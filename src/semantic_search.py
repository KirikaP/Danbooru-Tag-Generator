"""
语义搜索模块 - 基于Embedding API的Danbooru标签检索
使用硅基流动(SiliconFlow)的embedding API进行向量检索
"""

import os
import pickle
import re
import time
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import requests


class SemanticTagger:
    """基于Embedding API的Danbooru标签搜索引擎"""
    
    _instance = None  # 单例模式
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SemanticTagger, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self, csv_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        初始化语义标签引擎
        
        Args:
            csv_path: CSV数据库文件路径
            config: 配置字典，包含api_url, api_key, model等
        """
        if self.initialized:
            return
            
        self.csv_path = csv_path
        self.config = config or {}
        
        # 新的解耦配置结构
        embedding_cfg = self.config.get('embedding', {})
        reranker_cfg = self.config.get('reranker', {})
        llm_cfg = self.config.get('llm', {})
        
        # 兼容旧的配置格式（api_url, api_key 在顶层）
        self.embedding_api_url = embedding_cfg.get('api_url', self.config.get('api_url', llm_cfg.get('base_url', 'https://api.siliconflow.cn/v1')))
        self.embedding_api_key = embedding_cfg.get('api_key', self.config.get('api_key', llm_cfg.get('api_key', '')))
        self.embedding_model = embedding_cfg.get('model') or self.config.get('embedding_model') or self.config.get('model', 'Pro/BAAI/bge-m3')
        
        self.reranker_api_url = reranker_cfg.get('api_url', self.config.get('api_url', llm_cfg.get('base_url', 'https://api.siliconflow.cn/v1')))
        self.reranker_api_key = reranker_cfg.get('api_key', self.config.get('api_key', llm_cfg.get('api_key', '')))
        self.reranker_model = reranker_cfg.get('model') or self.config.get('reranker_model', 'Pro/BAAI/bge-reranker-v2-m3')
        
        # 数据
        self.df = None
        self.tags_data = []  # 原始标签列表（兼容旧接口）
        
        # 预计算的嵌入向量（从缓存加载）
        self.emb_en = None      # 英文名嵌入
        self.emb_cn = None      # 中文名嵌入  
        self.emb_wiki = None    # Wiki描述嵌入
        self.emb_cn_core = None # 中文核心词嵌入
        
        # 热度分数
        self.max_log_count = 15.0
        
        # 超时设置
        self.timeout = self.config.get('timeout', 60)
        
        # 停用词表
        self.stop_words = self._build_stop_words()
        
        # API会话（用于embedding）
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.embedding_api_key}',
            'Content-Type': 'application/json'
        })
        
        self.initialized = True
    
    def _tag_exists(self, tag_name: str) -> bool:
        """检查标签是否在数据库中"""
        if not hasattr(self, '_tag_names') or not self._tag_names:
            if self.df is not None:
                self._tag_names = set(self.df['name'].str.lower().tolist())
            else:
                return False
        return tag_name.lower() in self._tag_names

    def _request_with_retry(self, url: str, payload: dict, timeout: int = 60):
        """带重试的 POST 请求（指数退避）

        Args:
            url: 请求地址
            payload: JSON 请求体
            timeout: 单次超时秒数

        Returns:
            requests.Response 对象
        """
        max_retries: int = self.config.get('max_retries', 3)
        backoff_factor: float = self.config.get('backoff_factor', 2.0)
        retry_on_status: list = self.config.get('retry_on_status', [429, 500, 502, 503, 504])

        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                response = self.session.post(url, json=payload, timeout=timeout)
                if response.status_code in retry_on_status and attempt < max_retries:
                    wait = backoff_factor ** attempt
                    print(f"[SemanticTagger] 状态码 {response.status_code}，{wait:.1f} 秒后重试 "
                          f"(第{attempt + 1}/{max_retries}次)...")
                    time.sleep(wait)
                    continue
                return response
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = backoff_factor ** attempt
                    print(f"[SemanticTagger] 请求异常: {exc}，{wait:.1f} 秒后重试 "
                          f"(第{attempt + 1}/{max_retries}次)...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"API 请求在 {max_retries} 次重试后仍失败: {last_exc}")
    
    def _build_stop_words(self) -> set:
        """构建停用词表"""
        return {
            # 标点符号
            ',', '.', ':', ';', '?', '!', '"', "'", '`',
            '(', ')', '[', ']', '{', '}', '<', '>',
            '-', '_', '=', '+', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '~',
            '，', '。', '：', '；', '？', '！', '“', '”', '‘', '’',
            '（', '）', '【', '】', '《', '》', '、', '…', '—', '·',
            ' ', '\t', '\n', '\r',
            # 助词和虚词
            '的', '地', '得', '了', '着', '过',
            '是', '为', '被', '给', '把', '让', '由',
            '在', '从', '自', '向', '往', '对', '于',
            '和', '与', '及', '或', '且', '而', '但', '并', '即', '又', '也',
            # 感叹词
            '啊', '吗', '吧', '呢', '噢', '哦', '哈', '呀', '哇',
            # 代词
            '我', '你', '他', '她', '它', '我们', '你们', '他们',
            '这', '那', '此', '其', '谁', '啥', '某', '每',
            '这个', '那个', '这些', '那些', '这里', '那里',
            # 量词
            '个', '位', '只', '条', '张', '幅', '件', '套', '双', '对', '副',
            '种', '类', '群', '些', '点', '份', '部', '名',
            # 副词
            '很', '太', '更', '最', '挺', '特', '好', '真',
            # 数词
            '一', '一个', '一种', '一下', '一点', '一些',
            # 动词
            '有', '无', '非', '没', '不'
        }
    
    def load(self):
        """加载数据（延迟加载嵌入缓存）"""
        # 只加载CSV数据，不立即构建缓存
        # 构建标签名称集合用于快速查找
        self._tag_names = set()
        self._load_csv()
        
        # 构建标签名称集合（用于验证）
        if self.df is not None:
            self._tag_names = set(self.df['name'].str.lower().tolist())
        
        # 尝试加载缓存（支持两种格式）
        cache_path = self._get_cache_path()
        new_cache_path = "cache/embeddings_cache.pkl"
        
        # 优先尝试新格式的缓存
        if os.path.exists(new_cache_path):
            try:
                with open(new_cache_path, 'rb') as f:
                    data = pickle.load(f)
                    # 新格式: 单一组合嵌入
                    combined_emb = np.array(data['embeddings'], dtype=np.float32)
                    self.emb_en = combined_emb  # 使用组合嵌入代替英文嵌入
                    self.emb_cn = combined_emb  # 使用组合嵌入
                    self.emb_wiki = combined_emb
                    self.emb_cn_core = combined_emb
                    # 加载标签数据
                    self._cache_names = data.get('names', [])
                    self._cache_cn_names = data.get('cn_names', [])
                    self._cache_post_counts = data.get('post_counts', [])
                    self._cache_categories = data.get('categories', [])
                    self._cache_nsfw = data.get('nsfw', [])
                    # 更新df
                    if self.df is not None:
                        self.df['_cache_idx'] = range(len(self._cache_names))
                print(f"[SemanticTagger] Loaded NEW embeddings cache ({len(self._cache_names)} tags)")
                return
            except Exception as e:
                print(f"[SemanticTagger] New cache load failed: {e}")
        
        # 尝试旧格式缓存
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.emb_en = np.array(data['embeddings_en'], dtype=np.float32)
                    self.emb_cn = np.array(data['embeddings_cn'], dtype=np.float32)
                    self.emb_wiki = np.array(data.get('embeddings_wiki', np.zeros_like(self.emb_en)), dtype=np.float32)
                    self.emb_cn_core = np.array(data.get('embeddings_cn_core', np.zeros_like(self.emb_en)), dtype=np.float32)
                print(f"[SemanticTagger] Loaded OLD embeddings cache")
                return
            except Exception as e:
                print(f"[SemanticTagger] Old cache load failed: {e}")
        
        print(f"[SemanticTagger] Ready (Embedding API: {self.embedding_api_url}, Model: {self.embedding_model}; Reranker API: {self.reranker_api_url}, Model: {self.reranker_model})")
        print(f"[SemanticTagger] 未找到缓存，正在自动构建嵌入缓存...")
        self._auto_build_cache("cache/embeddings_cache.pkl")
        # 重新尝试加载
        if os.path.exists("cache/embeddings_cache.pkl"):
            with open("cache/embeddings_cache.pkl", 'rb') as f:
                data = pickle.load(f)
            combined_emb = np.array(data['embeddings'], dtype=np.float32)
            self.emb_en = combined_emb
            self.emb_cn = combined_emb
            self.emb_wiki = combined_emb
            self.emb_cn_core = combined_emb
            self._cache_names = data.get('names', [])
            self._cache_cn_names = data.get('cn_names', [])
            self._cache_post_counts = data.get('post_counts', [])
            self._cache_categories = data.get('categories', [])
            self._cache_nsfw = data.get('nsfw', [])
            print(f"[SemanticTagger] 缓存构建完成，已加载 ({len(self._cache_names)} 个标签)")
    
    def _load_csv(self):
        """加载CSV数据"""
        encodings = ['utf-8', 'gbk', 'gb18030']
        for enc in encodings:
            try:
                self.df = pd.read_csv(self.csv_path, dtype=str, encoding=enc).fillna("")
                break
            except:
                continue
        
        # 预处理数据
        if 'post_count' not in self.df.columns:
            self.df['post_count'] = 0
        self.df['post_count'] = pd.to_numeric(self.df['post_count'], errors='coerce').fillna(0)
        
        if 'cn_name' not in self.df.columns:
            self.df['cn_name'] = ""
        if 'wiki' not in self.df.columns:
            self.df['wiki'] = ""
        if 'name' not in self.df.columns:
            raise ValueError("CSV missing 'name' column")
        
        # 提取中文核心词
        self.df['cn_core'] = self.df['cn_name'].str.split(',', n=1).str[0].str.strip()
        
        # 更新标签数据
        self.tags_data = self.df.to_dict('records')
        self.max_log_count = float(np.log1p(self.df['post_count'].max()))
    
    def _get_cache_path(self) -> str:
        """获取缓存文件路径"""
        cache_dir = os.path.dirname(self.csv_path) if self.csv_path else '.'
        base_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        # 嵌入缓存与模型相关
        model_suffix = self.embedding_model.replace('/', '_')
        return os.path.join(cache_dir, f"{base_name}_emb_{model_suffix}.pkl")
    
    def _get_embedding(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        调用Embedding API获取文本向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            numpy数组，shape为 (len(texts), embedding_dim)
        """
        all_embeddings = []
        import time
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            
            payload = {
                "model": self.embedding_model,
                "input": batch,
                "encoding_format": "float"
            }
            
            try:
                batch_start = time.time()
                response = self._request_with_retry(
                    f"{self.embedding_api_url}/embeddings",
                    payload=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                batch_time = time.time() - batch_start
                
                # 提取embedding向量
                embeddings = [item['embedding'] for item in result['data']]
                all_embeddings.extend(embeddings)
                
                # 每个批次都输出进度
                if len(texts) > 100:  # 只有文本较多时才输出进度
                    print(f"    进度: [{batch_idx+1}/{total_batches}] 已编码 {min(i + batch_size, len(texts))}/{len(texts)} 个文本 (耗时: {batch_time:.2f}秒)")
                    
            except Exception as e:
                print(f"[SemanticTagger] API error: {e}")
                raise
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def _auto_build_cache(self, save_path: str):
        """自动构建嵌入缓存（新格式），加载全部标签并进行嵌入"""
        if self.df is None:
            print("[SemanticTagger] 数据未加载，无法构建缓存")
            return

        print(f"[SemanticTagger] 正在对 {len(self.df)} 个标签进行嵌入编码，请稍候...")

        texts = []
        for _, row in self.df.iterrows():
            name = row.get('name', '')
            cn_name = row.get('cn_name', '')
            text = f"{name} {cn_name.replace(',', ' ')}".strip() if cn_name else name
            texts.append(text)

        try:
            embeddings = self._get_embedding(texts, batch_size=32)
        except Exception as e:
            print(f"[SemanticTagger] 嵌入编码失败: {e}")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cache_data = {
            'embeddings': embeddings.tolist(),
            'names': self.df['name'].tolist(),
            'cn_names': self.df['cn_name'].tolist(),
            'post_counts': self.df['post_count'].tolist(),
            'categories': self.df['category'].tolist() if 'category' in self.df.columns else ['0'] * len(self.df),
            'nsfw': self.df['nsfw'].tolist() if 'nsfw' in self.df.columns else ['0'] * len(self.df),
        }
        with open(save_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"[SemanticTagger] 嵌入缓存已保存到: {save_path}")

    def _build_embeddings_cache(self, save_path: str):
        """从CSV构建嵌入缓存（调用API）"""
        # 读取CSV
        encodings = ['utf-8', 'gbk', 'gb18030']
        for enc in encodings:
            try:
                self.df = pd.read_csv(self.csv_path, dtype=str, encoding=enc).fillna("")
                break
            except:
                continue
        
        # 预处理数据
        if 'post_count' not in self.df.columns:
            self.df['post_count'] = 0
        self.df['post_count'] = pd.to_numeric(self.df['post_count'], errors='coerce').fillna(0)
        
        if 'cn_name' not in self.df.columns:
            self.df['cn_name'] = ""
        if 'wiki' not in self.df.columns:
            self.df['wiki'] = ""
        if 'name' not in self.df.columns:
            raise ValueError("CSV missing 'name' column")
        
        # 提取中文核心词（第一个逗号前的词）
        self.df['cn_core'] = self.df['cn_name'].str.split(',', n=1).str[0].str.strip()
        
        # 更新标签数据（兼容旧接口）
        self.tags_data = self.df.to_dict('records')
        
        print("[SemanticTagger] Encoding embeddings via API (this may take a while)...")
        
        # 批量编码嵌入向量
        self.emb_en = self._get_embedding(self.df['name'].tolist(), batch_size=32)
        self.emb_cn = self._get_embedding(self.df['cn_name'].tolist(), batch_size=32)
        self.emb_wiki = self._get_embedding(self.df['wiki'].tolist(), batch_size=32)
        self.emb_cn_core = self._get_embedding(self.df['cn_core'].tolist(), batch_size=32)
        
        # 保存缓存
        cache_data = {
            'df': self.df,
            'embeddings_en': self.emb_en.tolist(),
            'embeddings_cn': self.emb_cn.tolist(),
            'embeddings_wiki': self.emb_wiki.tolist(),
            'embeddings_cn_core': self.emb_cn_core.tolist(),
            'max_log_count': float(np.log1p(self.df['post_count'].max()))
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        self.max_log_count = cache_data['max_log_count']
        print(f"[SemanticTagger] Embeddings cache built: {save_path}")
    
    def _build_jieba_dict(self):
        """留作兼容性（已移除jieba）"""
        pass
    
    def _smart_split(self, text: str) -> List[str]:
        """
        智能分词：基于规则的分词（不依赖jieba）
        中文按标点和空格分割，英文按标点分割
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的token列表
        """
        tokens = []
        
        # 按中英文混合分割
        chunks = re.split(r'([\u4e00-\u9fa5]+)', text)
        
        for chunk in chunks:
            if not chunk.strip():
                continue
            
            # 中文：按标点和空格分割
            if re.match(r'[\u4e00-\u9fa5]+', chunk):
                # 移除punctuation后按空格分割
                cleaned = re.sub(r'[，。！？；：''""【】、·…—～]', ' ', chunk)
                parts = [p.strip() for p in cleaned.split() if p.strip()]
                tokens.extend(parts)
            else:
                # 英文：按标点分割
                parts = re.sub(r'[,()\[\]{}:;!?"\'\.\-~]', ' ', chunk).split()
                tokens.extend([p for p in parts if p and not p.isdigit()])
        
        return tokens
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        # a: (n, d), b: (m, d) -> (n, m)
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(a, b.T) / (a_norm * b_norm.T + 1e-8)
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        limit: int = 80,
        popularity_weight: float = 0.15
    ) -> Tuple[str, List[Dict]]:
        """
        语义搜索标签
        
        Args:
            query: 用户查询文本
            top_k: 每个分词查询的Top-K结果
            limit: 返回结果数量限制
            popularity_weight: 热度权重 (0-1)
            
        Returns:
            (标签字符串, 详细结果列表)
        """
        # 如果没有预计算的嵌入，则实时调用API进行检索
        if self.emb_en is None:
            return self._realtime_search(query, top_k, limit, popularity_weight)
        
        return self._cache_search(query, top_k, limit, popularity_weight)
    
    def _realtime_search(
        self, 
        query: str, 
        top_k: int = 5, 
        limit: int = 30,
        popularity_weight: float = 0.15
    ) -> Tuple[str, List[Dict]]:
        """
        直接语义搜索：使用完整query跳过中文分词问题
        1. 编码完整query
        2. 编码所有有效标签
        3. 直接计算相似度取Top结果
        """
        # 懒加载CSV数据
        if self.df is None:
            if self.csv_path:
                self._load_csv()
            if self.df is None:
                return "Error: No tags loaded.", []
        
        print(f"[SemanticTagger] 语义搜索阶段...")
        
        # 过滤出有效标签（非NSFW，非特殊分类）
        valid_tags = []
        for idx, row in self.df.iterrows():
            if int(row.get('nsfw', 0)) == 1:
                continue
            if int(row.get('category', 0)) == 4:
                continue
            valid_tags.append({
                'tag': row['name'],
                'cn_name': row['cn_name'],
                'post_count': row['post_count'],
                'category': row.get('category', '0'),
                'nsfw': row.get('nsfw', '0')
            })
        
        if not valid_tags:
            return self._fallback_search(query, limit)
        
        # 限制编码数量以避免API超时，默认最多500个
        max_encode = self.config.get('max_encode_tags', 500)
        if len(valid_tags) > max_encode:
            # 按热度排序，取前max_encode个
            valid_tags = sorted(valid_tags, key=lambda x: int(x['post_count']), reverse=True)[:max_encode]
        
        try:
            return self._direct_semantic_match(query, valid_tags, limit, popularity_weight)
        except Exception as e:
            print(f"[SemanticTagger] 语义搜索失败: {e}")
            return self._fallback_search(query, limit)
    
    def _direct_semantic_match(
        self,
        query: str,
        candidates: List[Dict],
        limit: int = 30,
        popularity_weight: float = 0.15
    ) -> Tuple[str, List[Dict]]:
        """直接语义匹配：编码 query 变体和所有候选标签，取最大相似度"""
        queries = self._extract_queries(query)
        n_queries = len(queries)

        # 准备编码文本：先放所有查询变体，再放候选文本
        candidate_texts = [f"{c['tag']} {c['cn_name']}" for c in candidates]
        all_texts = queries + candidate_texts
        
        # 批量编码
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            embeddings = self._get_embedding(batch)
            all_embeddings.append(embeddings)
        
        all_embeddings = np.vstack(all_embeddings)
        query_emb = all_embeddings[:n_queries]           # (Q, D)
        candidate_embs = all_embeddings[n_queries:]      # (C, D)
        
        # 计算相似度：多个查询变体取最大值 → (C,)
        sim_matrix = self._cosine_similarity(query_emb, candidate_embs)  # (Q, C)
        similarities = np.max(sim_matrix, axis=0)  # (C,)
        
        # 综合评分
        final_results = {}
        max_log_count = np.log1p(np.max([int(c['post_count']) for c in candidates])) if candidates else 1
        
        for i, cand in enumerate(candidates):
            semantic_score = similarities[i]
            pop_score = np.log1p(int(cand['post_count'])) / max_log_count if max_log_count > 0 else 0
            
            final_score = semantic_score * (1 - popularity_weight) + pop_score * popularity_weight
            
            final_results[cand['tag']] = {
                'tag': cand['tag'],
                'final_score': float(final_score),
                'semantic_score': float(semantic_score),
                'source': query[:20],
                'cn_name': cand['cn_name'],
                'layer': 'DirectMatch',
                'category': cand['category'],
                'nsfw': cand['nsfw'],
                'post_count': cand['post_count']
            }
        
        # 排序取Top
        sorted_tags = sorted(final_results.values(), key=lambda x: x['final_score'], reverse=True)
        
        # 过滤低相似度
        similarity_threshold = self.config.get('similarity_threshold', 0.5)
        final_list = [t for t in sorted_tags[:limit] if t.get('semantic_score', 0) >= similarity_threshold]
        
        print(f"[SemanticTagger] 直接语义匹配完成，得到 {len(final_list)} 个标签")
        
        tags_string = ", ".join([item['tag'] for item in final_list])
        return tags_string, final_list
    
    def _keyword_filter(self, query: str, limit: int = 200) -> List[Dict]:
        """关键词快速筛选候选标签"""
        candidates = []
        query_lower = query.lower()
        
        # 预先提取查询中的关键概念
        keywords = self._smart_split(query)
        kw_set = set(keywords)
        
        # 特殊中文映射（用于补充cn_name可能遗漏的变体）
        special_map = {
            '黄': 'blonde', '金': 'blonde',
            '蓝眼': 'blue_eyes', '蓝眸': 'blue_eyes',
            '红眼': 'red_eyes', '红眸': 'red_eyes',
            '绿眼': 'green_eyes', '绿眸': 'green_eyes',
            '女仆': 'maid',
            '猫耳': 'cat_ears', '猫娘': 'cat_girl',
            '长发': 'long_hair', '短发': 'short_hair',
            '校服': 'school_uniform', '制服': 'school_uniform',
            '比基尼': 'bikini', '泳装': 'swimsuit',
            '和服': 'kimono', '浴衣': 'yukata',
            '图书馆': 'library', '书店': 'bookstore',
            '樱花': 'cherry_blossoms',
            '星空': 'starry_sky',
        }
        
        for cn_word, tag_keyword in special_map.items():
            if cn_word in query:
                kw_set.add(tag_keyword)
        
        for idx, row in self.df.iterrows():
            tag_name = row['name']
            cn_name = row['cn_name']
            
            # 跳过NSFW和特殊标签
            if int(row.get('nsfw', 0)) == 1:
                continue
            if int(row.get('category', 0)) == 4:
                continue
            
            score = 0
            
            # 核心标签强制加入（如1girl, 1boy等）
            if tag_name in ['1girl', '1boy', 'solo', 'cat_girl', 'cat_ears']:
                if any(kw in tag_name.lower() or kw in cn_name.lower() for kw in kw_set):
                    score += 50
            
            # 强制纳入常见的必须标签（基于查询中的关键概念）
            must_have_tags = ['1girl', '1boy', 'solo']
            
            # 根据查询中的概念强制添加相关标签
            if '女' in query or 'girl' in kw_set:
                must_have_tags.append('1girl')
            if '男' in query or 'boy' in kw_set:
                must_have_tags.append('1boy')
            if any(w in query for w in ['红', '赤']):
                must_have_tags.extend(['red_hair', 'red_eyes'])
            if '绿' in query:
                must_have_tags.extend(['green_eyes'])
            if '蓝' in query:
                must_have_tags.extend(['blue_eyes', 'blue_hair'])
            if '黄' in query or '金' in query:
                must_have_tags.extend(['blonde_hair', 'yellow_hair'])
            if '白' in query:
                must_have_tags.append('white_hair')
            if '黑' in query:
                must_have_tags.append('black_hair')
            if '紫' in query:
                must_have_tags.extend(['purple_hair', 'purple_eyes'])
            if '粉' in query:
                must_have_tags.extend(['pink_hair', 'pink_eyes'])
            if '猫' in query:
                must_have_tags.extend(['cat', 'cat_ears', 'cat_girl'])
            if '图' in query and '书' in query:
                must_have_tags.append('library')
            
            for must_tag in must_have_tags:
                if tag_name == must_tag:
                    score += 100  # 强制高分
            
            # 中文匹配
            if cn_name:
                aliases = cn_name.split(',')
                for alias in aliases:
                    alias = alias.strip()
                    if len(alias) >= 2:
                        if alias in query:
                            score += 10
                        # 分词匹配
                        for kw in keywords:
                            if kw in alias:
                                score += 2
            
            # 英文匹配（tag_name）
            tag_name_lower = tag_name.lower().replace('_', ' ')
            if tag_name_lower in query_lower:
                score += 15
            
            # 基于关键概念的匹配
            for kw in kw_set:
                if kw in tag_name_lower or kw in tag_name:
                    score += 5
                if cn_name and kw in cn_name.lower():
                    score += 5
            
            # 热度加分
            post_count = int(row.get('post_count', 0))
            if post_count > 10000:
                score += 1
            if post_count > 100000:
                score += 2
            
            if score > 0:
                candidates.append({
                    'idx': idx,
                    'tag': tag_name,
                    'cn_name': cn_name,
                    'score': score,
                    'post_count': post_count,
                    'category': row.get('category', '0'),
                    'nsfw': row.get('nsfw', '0')
                })
        
        # 按分数排序，取top候选
        candidates.sort(key=lambda x: (x['score'], x['post_count']), reverse=True)
        return candidates[:limit]
    
    def _semantic_rerank(
        self, 
        query: str, 
        candidates: List[Dict],
        top_k: int = 5, 
        limit: int = 30,
        popularity_weight: float = 0.15
    ) -> Tuple[str, List[Dict]]:
        """语义搜索重排序候选标签"""
        import time
        
        # 提取候选的文本用于编码
        candidate_texts = [f"{c['tag']} {c['cn_name']}" for c in candidates]
        
        # 添加查询文本
        all_texts = [query] + candidate_texts
        
        try:
            embeddings = self._get_embedding(all_texts)
            query_emb = embeddings[0:1]  # (1, d)
            candidate_embs = embeddings[1:]  # (n, d)
            
            # 计算相似度
            similarities = self._cosine_similarity(query_emb, candidate_embs)[0]  # (n,)
            
            # 排序和后处理
            final_results = {}
            
            for i, cand in enumerate(candidates):
                semantic_score = similarities[i]
                pop_score = np.log1p(float(cand['post_count'])) / self.max_log_count
                
                # 综合评分 = 关键词分数*0.3 + 语义分*0.5 + 热度分*0.2
                keyword_norm = cand['score'] / 20.0  # 归一化
                final_score = (
                    keyword_norm * 0.3 + 
                    semantic_score * 0.5 + 
                    pop_score * popularity_weight
                )
                
                tag_name = cand['tag']
                if tag_name not in final_results or final_score > final_results[tag_name]['final_score']:
                    final_results[tag_name] = {
                        'tag': tag_name,
                        'final_score': float(final_score),
                        'semantic_score': float(semantic_score),
                        'keyword_score': float(keyword_norm),
                        'source': query[:20],
                        'cn_name': cand['cn_name'],
                        'layer': 'SemanticRerank',
                        'category': cand['category'],
                        'nsfw': cand['nsfw'],
                        'post_count': cand['post_count']
                    }
            
            # 排序
            sorted_tags = sorted(final_results.values(), key=lambda x: x['final_score'], reverse=True)
            
            # 只保留在数据库中存在的标签，且相似度 >= 阈值
            similarity_threshold = self.config.get('similarity_threshold', 0.5)
            valid_tags = [t for t in sorted_tags[:limit] if self._tag_exists(t['tag']) and t.get('semantic_score', 0) >= similarity_threshold]
            
            print(f"[SemanticTagger]   得到 {len(valid_tags)} 个有效标签")
            
            final_list = valid_tags
            tags_string = ", ".join([item['tag'] for item in final_list])
            
            return tags_string, final_list
            
        except Exception as e:
            print(f"[SemanticTagger] Semantic rerank failed: {e}")
            # 回退到只用关键词分数（只保留有效标签）
            valid_candidates = [c for c in candidates if self._tag_exists(c['tag'])]
            fallback = {c['tag']: c for c in valid_candidates[:limit]}
            final_list = list(fallback.values())
            tags_string = ", ".join([item['tag'] for item in final_list])
            print(f"[SemanticTagger]   回退搜索，保留 {len(final_list)} 个标签")
            return tags_string, final_list
    
    def _extract_queries(self, query: str) -> List[str]:
        """从描述中提取多个查询变体（保留完整句 + 中文部分 + 英文部分）"""
        queries = [query.strip()]
        if ' / ' in query:
            parts = [p.strip() for p in query.split(' / ') if p.strip()]
            for p in parts:
                if p not in queries:
                    queries.append(p)
        return queries

    def _call_reranker_api(self, query: str, documents: List[str], top_n: int) -> List[tuple]:
        """
        调用 cross-encoder reranker API，返回 [(index, relevance_score), ...] 按分数降序
        """
        payload = {
            "model": self.reranker_model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False
        }
        headers = {
            'Authorization': f'Bearer {self.reranker_api_key}',
            'Content-Type': 'application/json'
        }
        timeout = self.config.get('timeout', 60)
        max_retries = self.config.get('max_retries', 3)
        backoff = self.config.get('backoff_factor', 2.0)

        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.reranker_api_url}/rerank",
                    headers=headers, json=payload, timeout=timeout
                )
                resp.raise_for_status()
                results = resp.json().get('results', [])
                return [(r['index'], r['relevance_score']) for r in results]
            except Exception as e:
                if attempt < max_retries:
                    wait = backoff ** attempt
                    print(f"[SemanticTagger] Reranker retry {attempt+1}/{max_retries} after {wait:.1f}s: {e}")
                    import time as _t; _t.sleep(wait)
                else:
                    raise
        return []

    def _cache_search(
        self, 
        query: str, 
        top_k: int = 5, 
        limit: int = 30,
        popularity_weight: float = 0.15
    ) -> Tuple[str, List[Dict]]:
        """
        使用缓存进行直接语义匹配（跳过中文分词）
        支持中英双查询合并 + reranker 精排
        """
        # 提取多查询变体（整句 / 中文部分 / 英文部分）
        queries = self._extract_queries(query)
        try:
            query_embeddings = self._get_embedding(queries)  # (Q, D)
        except Exception as e:
            print(f"[SemanticTagger] Query encoding failed: {e}")
            return self._fallback_search(query, limit)
        
        # 语义检索（多层）—— 对每个查询变体分别计算，取最大相似度
        def max_sim(emb_matrix):
            sims = self._cosine_similarity(query_embeddings, emb_matrix)  # (Q, N)
            return np.max(sims, axis=0, keepdims=True)  # (1, N)

        sim_en    = max_sim(self.emb_en)
        sim_cn    = max_sim(self.emb_cn)
        sim_wiki  = max_sim(self.emb_wiki)
        sim_cn_core = max_sim(self.emb_cn_core)
        
        # 获取Top-K结果索引
        def get_topk_indices(sim_matrix, k):
            return np.argsort(-sim_matrix, axis=1)[:, :k]
        
        hits_en    = get_topk_indices(sim_en,     limit)
        hits_cn    = get_topk_indices(sim_cn,     limit)
        hits_wiki  = get_topk_indices(sim_wiki,   limit)
        hits_cn_core = get_topk_indices(sim_cn_core, limit)
        
        # 合并四层结果
        final_results = {}
        
        for layer_name, hits in [('EN', hits_en[0]), ('CN', hits_cn[0]), ('Wiki', hits_wiki[0]), ('Core', hits_cn_core[0])]:
            for idx in hits:
                score = {
                    'EN': sim_en[0, idx],
                    'CN': sim_cn[0, idx],
                    'Wiki': sim_wiki[0, idx],
                    'Core': sim_cn_core[0, idx]
                }[layer_name]
                
                row = self.df.iloc[idx]
                tag_name = row['name']
                
                # 热度分数
                pop_score = np.log1p(float(row['post_count'])) / self.max_log_count
                
                # 综合评分 = 语义分*(1-热度权重) + 热度分*热度权重
                final_score = (score * (1 - popularity_weight)) + (pop_score * popularity_weight)
                
                # 更新最优结果
                if tag_name not in final_results or final_score > final_results[tag_name]['final_score']:
                    final_results[tag_name] = {
                        'tag': tag_name,
                        'final_score': float(final_score),
                        'semantic_score': float(score),
                        'source': query[:20],
                        'cn_name': row['cn_name'],
                        'layer': layer_name,
                        'category': row.get('category', '0'),
                        'nsfw': row.get('nsfw', '0'),
                        'post_count': row.get('post_count', '0')
                    }
        
        # 5. 排序并取Top结果
        sorted_tags = sorted(final_results.values(), key=lambda x: x['final_score'], reverse=True)
        
        # 过滤NSFW、特殊标签和低相似度标签
        similarity_threshold = self.config.get('similarity_threshold', 0.5)
        filtered_tags = []
        for tag in sorted_tags:
            # 排除NSFW
            if int(tag.get('nsfw', 0)) == 1:
                continue
            # 排除分类4（特殊角色）
            if int(tag.get('category', 0)) == 4:
                continue
            # 排除相似度低于阈值的标签
            if tag.get('semantic_score', 0) < similarity_threshold:
                continue
            filtered_tags.append(tag)
        
        pre_rerank = filtered_tags[:limit * 2]  # 取多一些候选给reranker

        # 6. Reranker 精排（如果配置了 reranker 且候选数量足够）
        reranker_cfg = self.config.get('reranker', {})
        use_reranker = reranker_cfg.get('enabled', True) and self.reranker_api_key and len(pre_rerank) > 0
        if use_reranker:
            # 优先用英文部分查询做 reranker（语义更精准）
            rerank_query = queries[-1] if len(queries) > 1 else query
            try:
                documents = [f"{t['tag']} {t['cn_name']}" for t in pre_rerank]
                ranked = self._call_reranker_api(rerank_query, documents, top_n=limit)
                # 按 reranker 分数重新排列
                idx_score = {r[0]: r[1] for r in ranked}
                for i, t in enumerate(pre_rerank):
                    if i in idx_score:
                        t['reranker_score'] = idx_score[i]
                        # 融合 reranker 分数：原分 * 0.3 + reranker_norm * 0.7
                        t['final_score'] = t['final_score'] * 0.3 + idx_score[i] * 0.7
                pre_rerank.sort(key=lambda x: x['final_score'], reverse=True)
                print(f"[SemanticTagger] Reranker 精排完成")
            except Exception as e:
                print(f"[SemanticTagger] Reranker 失败，跳过: {e}")

        final_list = pre_rerank[:limit]
        print(f"[SemanticTagger] 搜索完成，得到 {len(final_list)} 个标签")
        
        # 7. 生成标签字符串
        tags_string = ", ".join([item['tag'] for item in final_list])
        
        return tags_string, final_list
    
    def _fallback_search(self, query: str, limit: int = 30) -> Tuple[str, List[Dict]]:
        """
        回退搜索：使用简单关键词匹配
        """
        if not self.tags_data:
            return "Error: No tags loaded.", []
        
        query_lower = query.lower()
        results = []
        
        for tag in self.tags_data:
            tag_name = tag.get('name', '')
            cn_name = tag.get('cn_name', '')
            wiki = tag.get('wiki', '')
            
            # 跳过NSFW
            if int(tag.get('nsfw', 0)) == 1:
                continue
            
            # 跳过分类4
            if int(tag.get('category', 0)) == 4:
                continue
            
            # 简单匹配
            score = 0
            if query_lower in tag_name.lower():
                score += 10
            if query_lower in cn_name.lower():
                score += 20
            if query_lower in wiki.lower():
                score += 5
            
            if score > 0:
                post_count = int(tag.get('post_count', 0))
                pop_score = np.log1p(post_count) / self.max_log_count if self.max_log_count > 0 else 0
                
                results.append({
                    'tag': tag_name,
                    'final_score': score + pop_score * 0.1,
                    'semantic_score': score,
                    'source': query,
                    'cn_name': cn_name,
                    'layer': 'Fallback',
                    'category': tag.get('category', '0'),
                    'nsfw': tag.get('nsfw', '0'),
                    'post_count': post_count
                })
        
        results.sort(key=lambda x: x['final_score'], reverse=True)
        final_list = results[:limit]
        
        tags_string = ", ".join([item['tag'] for item in final_list])
        
        return tags_string, final_list


def create_semantic_tagger(csv_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> SemanticTagger:
    """
    工厂函数：创建语义标签引擎实例
    
    Args:
        csv_path: CSV数据库路径
        config: 配置字典
        
    Returns:
        SemanticTagger实例
    """
    tagger = SemanticTagger(csv_path, config)
    tagger.load()
    return tagger