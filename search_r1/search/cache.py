import redis
import json
import os
import datetime
import time
import glob
from typing import Optional, Any, List, Tuple, Dict
import hashlib

class SearchCache:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, 
                 dataset_name: str = "default", max_load_entries: int = 600000):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        
        # 清理Redis缓存
        self.redis_client.flushdb()
        print(f"已清理Redis数据库 {db}")
        
        self.local_cache_dir = os.path.join(os.path.dirname(__file__), "local_cache")
        os.makedirs(self.local_cache_dir, exist_ok=True)
        
        # 使用数据集名称创建专用文件夹
        self.dataset_name = dataset_name
        self.dataset_cache_dir = os.path.join(self.local_cache_dir, dataset_name)
        self.batch_cache_dir = os.path.join(self.dataset_cache_dir, "cache")
        os.makedirs(self.batch_cache_dir, exist_ok=True)

        
        # 最大加载条目数
        self.max_load_entries = max_load_entries
        
        # 增量缓存文件的前缀
        # self.cache_file_prefix = "search_cache_630_nocache_wiki18_e5_topk1"
        self.cache_file_prefix = "search_cache_9_3_musique_e5_topk3"

        
        # 内存缓存
        self.memory_cache = {}
        self.memory_cache_size = 200000  # 内存中最多保存10000个查询结果
        
        # Redis缓存的最大大小和批处理大小
        self.redis_cache_max_size = 600000  # Redis最多存储60万条
        self.batch_save_size = 5000  # 每累积5000条保存一次
        
        # 新增条目计数
        self.new_entries_count = 0
        
        # 待保存的批量数据
        self.pending_batch = {}
        
        # 内存中保存查询和键的映射
        self.memory_query_map: Dict[str, str] = {}
        
        # 加载本地缓存到内存缓存
        loaded_entries = self._load_to_memory_cache()
        
        # 初始化更详细的统计数据
        self.stats = {
            "total_requests": 0,
            "redis_hits": 0,
            "local_hits": 0, 
            "misses": 0,
            "last_reset": time.time(),
            "redis_hit_rate": 0.0,
            "local_hit_rate": 0.0,
            "miss_rate": 0.0,
            "total_hit_rate": 0.0,
            "loaded_entries": loaded_entries,
            "max_load_entries": self.max_load_entries
        }
        
        self.new_stats_file()

    def _load_to_memory_cache(self) -> int:
        """从数据集目录中加载增量缓存文件到Redis缓存，返回加载的条目数"""
        entries_loaded = 0
        
        # 获取所有增量缓存文件，按时间戳排序（最新的先加载）
        cache_files = glob.glob(os.path.join(self.dataset_cache_dir, "cache", f"{self.cache_file_prefix}*.json"))
        cache_files.sort(key=os.path.getmtime, reverse=True)
        
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        file_cache = json.load(f)
                        
                        # 检查是否超过最大加载限制
                        remaining_capacity = self.max_load_entries - entries_loaded
                        if remaining_capacity <= 0:
                            break
                            
                        # 如果文件中的条目超过剩余容量，只加载部分
                        items_to_load = list(file_cache.items())
                        if len(items_to_load) > remaining_capacity:
                            items_to_load = items_to_load[:remaining_capacity]
                            
                        # 将数据加载到Redis缓存中
                        pipe = self.redis_client.pipeline()
                        for key, value in items_to_load:
                            pipe.set(key, json.dumps(value))
                            entries_loaded += 1
                        pipe.execute()
                            
                        # 如果已经达到最大加载限制，跳出循环
                        if entries_loaded >= self.max_load_entries:
                            break
                            
                except json.JSONDecodeError:
                    print(f"无法解析缓存文件: {cache_file}")
                    continue
                
        print(f"已从{self.dataset_name}数据集加载{entries_loaded}条缓存记录到Redis中")
        return entries_loaded

    def new_stats_file(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stats_dir = os.path.join(self.dataset_cache_dir)
        os.makedirs(self.stats_dir, exist_ok=True)
        self.stats_file = os.path.join(self.stats_dir, f"cache_stats_{timestamp}.json")
        return self.stats_file

    def update_stats(self, stats: dict):
        with open(self.stats_file, "w") as f:
            json.dump(stats, f,indent=2)
            
    def get_status(self):
        """获取缓存状态，增加了新条目计数和待保存批次大小的信息"""
        status = self.stats.copy()
        status["new_entries_count"] = self.new_entries_count
        status["pending_batch_size"] = len(self.pending_batch)
        status["redis_cache_max_size"] = self.redis_cache_max_size
        status["batch_save_size"] = self.batch_save_size
        status["loaded_entries"] = len(self.memory_cache)
        status["max_load_entries"] = self.max_load_entries
        return status
    
    def _get_cache_key(self, query: str) -> str:
        """生成缓存键"""
        return hashlib.md5(query.encode()).hexdigest()
        
            
    def _save_batch_to_local(self):
        """将积累的批量数据保存为增量文件并清空批量缓存"""
        if not self.pending_batch:
            return
            
        # 生成新的增量缓存文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        increment_file = os.path.join(self.batch_cache_dir, f"{self.cache_file_prefix}{timestamp}.json")
        
        # 保存当前批次到增量文件
        with open(increment_file, 'w') as f:
            json.dump(self.pending_batch, f,indent=2)
            
        # 重置批量缓存和计数器
        batch_size = len(self.pending_batch)
        self.pending_batch = {}
        self.new_entries_count = 0
        
        print(f"已保存{batch_size}条缓存记录到增量文件: {increment_file}")

    def get(self, query: str) -> Optional[Any]:
        """从缓存中获取数据，现在只检查内存缓存和Redis缓存"""
        # 增加请求计数
        self.stats["total_requests"] += 1
        
        cache_key = self._get_cache_key(query)
        
        # 1. 检查内存缓存
        if cache_key in self.memory_cache:
            self.stats["local_hits"] += 1
            self._update_hit_rates()
            return self.memory_cache[cache_key]
        
        # 2. 检查Redis缓存
        redis_data = self.redis_client.get(cache_key)
        if redis_data:
            data = json.loads(redis_data)
            self._update_memory_cache(cache_key, data)
            self.stats["redis_hits"] += 1
            self._update_hit_rates()
            return data

        # 缓存未命中
        self.stats["misses"] += 1
        self._update_hit_rates()
        return None

    def _update_hit_rates(self):
        """更新所有命中率统计"""
        if self.stats["total_requests"] > 0:
            total_hits = self.stats["redis_hits"] + self.stats["local_hits"]
            self.stats["redis_hit_rate"] = self.stats["redis_hits"] / self.stats["total_requests"]
            self.stats["local_hit_rate"] = self.stats["local_hits"] / self.stats["total_requests"]
            self.stats["miss_rate"] = self.stats["misses"] / self.stats["total_requests"]
            self.stats["total_hit_rate"] = total_hits / self.stats["total_requests"]
        
        # 减少频繁写入统计文件
        if self.stats["total_requests"] % 100 == 0:
            self.update_stats(self.stats)

    def _update_memory_cache(self, key: str, data: Any):
        """更新内存缓存"""
        if len(self.memory_cache) >= self.memory_cache_size:
            # 如果内存缓存已满，删除最旧的条目
            self.memory_cache.pop(next(iter(self.memory_cache)))
        self.memory_cache[key] = data

    def set(self, query: str, data: Any, expire_time: int = 3600) -> None:
        """将数据存入缓存"""
        cache_key = self._get_cache_key(query)
        
        # 1. 存入内存缓存
        self._update_memory_cache(cache_key, data)
        
        # 2. 检查Redis缓存大小
        redis_size = self.redis_client.dbsize()
        
        # 3. 如果Redis中的数据少于最大容量，则存入Redis
        if redis_size < self.redis_cache_max_size:
            self.redis_client.setex(
                cache_key,
                expire_time,
                json.dumps(data)
            )
        
        # 4. 将数据加入待保存的批量数据
        self.pending_batch[cache_key] = data
        self.new_entries_count += 1
        
        # 5. 如果积累了足够多的新条目，就批量保存到本地
        if self.new_entries_count >= self.batch_save_size:
            self._save_batch_to_local()