#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建语义搜索缓存脚本
将CSV中所有标签进行嵌入和重排，结果保存在本地
"""

import os
import pickle
import time
import json
import numpy as np
import pandas as pd
import requests
from pathlib import Path


# 配置 - 从config.json读取
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import get_config

config = get_config()
semantic_cfg = config.get('semantic_search', {})
llm_cfg = config.get('llm', config.get('api', {}))

CSV_PATH = "danbooruTags/tags_enhanced.csv"

# 新的解耦配置结构
embedding_cfg = semantic_cfg.get('embedding', {})
reranker_cfg = semantic_cfg.get('reranker', {})

# 兼容旧的配置格式
EMBEDDING_MODEL = embedding_cfg.get('model') or semantic_cfg.get('embedding_model') or semantic_cfg.get('model', 'Pro/BAAI/bge-m3')
EMBEDDING_API_URL = embedding_cfg.get('api_url', semantic_cfg.get('api_url', llm_cfg.get('base_url', 'https://api.siliconflow.cn/v1')))
EMBEDDING_API_KEY = embedding_cfg.get('api_key', semantic_cfg.get('api_key', llm_cfg.get('api_key', '')))

RERANKER_MODEL = reranker_cfg.get('model') or semantic_cfg.get('reranker_model', 'Pro/BAAI/bge-reranker-v2-m3')
RERANKER_API_URL = reranker_cfg.get('api_url', semantic_cfg.get('api_url', llm_cfg.get('base_url', 'https://api.siliconflow.cn/v1')))
RERANKER_API_KEY = reranker_cfg.get('api_key', semantic_cfg.get('api_key', llm_cfg.get('api_key', '')))

CACHE_DIR = "cache"
EMBED_CACHE_PATH = os.path.join(CACHE_DIR, "embeddings_cache.pkl")
RERANK_CACHE_PATH = os.path.join(CACHE_DIR, "rerank_cache.pkl")

BATCH_SIZE = 32  # 嵌入批次大小
CHECKPOINT_PATH = os.path.join(CACHE_DIR, "embedding_checkpoint.pkl")
MAX_RETRIES = 3
RETRY_DELAY = 2  # 重试延迟（秒）


def load_csv():
    """加载CSV数据库"""
    print("加载CSV数据库...")
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
    
    for enc in encodings:
        try:
            df = pd.read_csv(CSV_PATH, dtype=str, encoding=enc).fillna("")
            print(f"  加载成功，共 {len(df)} 个标签 (编码: {enc})")
            return df
        except Exception as e:
            print(f"  编码 {enc} 失败: {e}")
            continue
    
    raise Exception("无法加载CSV文件")


def save_checkpoint(start_idx, embeddings):
    """保存进度检查点"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    checkpoint = {
        'start_idx': start_idx,
        'embeddings': embeddings
    }
    with open(CHECKPOINT_PATH, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"  [检查点已保存: {start_idx}]")


def load_checkpoint():
    """加载进度检查点"""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"  [找到检查点，从 {checkpoint['start_idx']} 继续]")
        return checkpoint['start_idx'], checkpoint['embeddings']
    return 0, None


def get_embedding_with_retry(texts, batch_size=32, start_from=0, existing_embeddings=None):
    """调用嵌入API获取文本嵌入（带重试和断点续传）"""
    session = requests.Session()
    session.headers.update({
        'Authorization': f'Bearer {EMBEDDING_API_KEY}',
        'Content-Type': 'application/json'
    })
    
    all_embeddings = []
    
    # 如果有检查点，加载已有的embeddings
    if start_from > 0 and existing_embeddings is not None:
        all_embeddings = list(existing_embeddings)
        print(f"  继续从位置 {start_from} 开始...")
    
    total = len(texts)
    
    for i in range(start_from, total, batch_size):
        batch = texts[i:i+batch_size]
        batch_idx = i
        
        payload = {
            "model": EMBEDDING_MODEL,
            "input": batch,
            "encoding_format": "float"
        }
        
        # 重试逻辑
        for retry in range(MAX_RETRIES):
            try:
                resp = session.post(f"{EMBEDDING_API_URL}/embeddings", json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                
                embeddings = [item['embedding'] for item in data['data']]
                all_embeddings.extend(embeddings)
                
                current = min(i+batch_size, total)
                print(f"  已处理 {current}/{total} 个标签")
                
                # 每500个标签保存一次检查点
                if (i + batch_size) % 500 == 0 or i + batch_size >= total:
                    save_checkpoint(i + batch_size, all_embeddings)
                
                break  # 成功，跳出重试循环
                
            except Exception as e:
                error_msg = str(e)
                if '502' in error_msg or 'Bad Gateway' in error_msg:
                    print(f"  [警告] API错误 (502)，{RETRY_DELAY}秒后重试... ({retry+1}/{MAX_RETRIES})")
                elif '429' in error_msg or 'rate' in error_msg.lower():
                    wait_time = RETRY_DELAY * (retry + 1) * 2
                    print(f"  [警告] 速率限制，{wait_time}秒后重试... ({retry+1}/{MAX_RETRIES})")
                else:
                    print(f"  嵌入失败: {e}，{RETRY_DELAY}秒后重试... ({retry+1}/{MAX_RETRIES})")
                
                if retry < MAX_RETRIES - 1:
                    if '429' in error_msg:
                        time.sleep(RETRY_DELAY * (retry + 1) * 2)
                    else:
                        time.sleep(RETRY_DELAY)
                else:
                    # 最后一次重试失败，保存检查点再抛出异常
                    save_checkpoint(i, all_embeddings)
                    raise Exception(f"API调用失败，已保存检查点: {error_msg}")
    
    return np.array(all_embeddings, dtype=np.float32)


def rerank(query, texts, top_n=10):
    """调用重排API"""
    session = requests.Session()
    session.headers.update({
        'Authorization': f'Bearer {RERANKER_API_KEY}',
        'Content-Type': 'application/json'
    })
    
    payload = {
        "model": RERANKER_MODEL,
        "query": query,
        "documents": texts,
        "top_n": top_n,
        "return_documents": False
    }
    
    try:
        resp = session.post(f"{RERANKER_API_URL}/rerank", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        # 返回重排后的索引和分数
        results = data.get('results', [])
        return [(r['index'], r['relevance_score']) for r in results]
        
    except Exception as e:
        print(f"  重排失败: {e}")
        raise


def build_embeddings_cache(df):
    """构建嵌入缓存（支持断点续传）"""
    print("\n" + "="*50)
    print("步骤1: 嵌入所有标签...")
    print("="*50)
    
    # 检查是否有完整缓存
    if os.path.exists(EMBED_CACHE_PATH):
        with open(EMBED_CACHE_PATH, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"  找到现有嵌入缓存，共 {len(cache_data['names'])} 个标签")
        print(f"  如需重新构建，请删除 {EMBED_CACHE_PATH}")
        embeddings = np.array(cache_data['embeddings'], dtype=np.float32)
        return embeddings, df
    
    # 准备文本（英文名 + 中文名）
    texts = []
    for _, row in df.iterrows():
        name = row.get('name', '')
        cn_name = row.get('cn_name', '')
        
        # 组合文本
        if cn_name:
            text = f"{name} {cn_name.replace(',', ' ')}"
        else:
            text = name
        
        texts.append(text)
    
    print(f"共 {len(texts)} 个标签需要嵌入")
    print(f"API模型: {EMBEDDING_MODEL}")
    print(f"每批处理: {BATCH_SIZE} 个")
    
    # 检查断点续传
    start_idx = 0
    existing_embeddings = None
    if os.path.exists(CHECKPOINT_PATH):
        start_idx, existing_embeddings = load_checkpoint()
        if start_idx >= len(texts):
            print("  检查点已表示完成，直接加载")
            embeddings = existing_embeddings
            # 保存完整缓存
            cache_data = {
                'embeddings': embeddings.tolist(), # type: ignore
                'names': df['name'].tolist(),
                'cn_names': df['cn_name'].tolist(),
                'post_counts': df['post_count'].tolist() if 'post_count' in df.columns else ['0'] * len(df),
                'categories': df['category'].tolist() if 'category' in df.columns else ['0'] * len(df),
                'nsfw': df['nsfw'].tolist() if 'nsfw' in df.columns else ['0'] * len(df),
            }
            with open(EMBED_CACHE_PATH, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"嵌入缓存已保存到: {EMBED_CACHE_PATH}")
            # 删除检查点
            try:
                os.remove(CHECKPOINT_PATH)
            except:
                pass
            return embeddings, df
    
    # 计算预估成本和时间
    estimated_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"预估批次: {estimated_batches}")
    print(f"预估API调用: ~{estimated_batches} 次")
    print(f"开始时间: {time.strftime('%H:%M:%S')}")
    
    # 开始嵌入
    start_time = time.time()
    embeddings = get_embedding_with_retry(texts, BATCH_SIZE, start_idx, existing_embeddings)
    elapsed = time.time() - start_time
    
    print(f"\n嵌入完成! 耗时: {elapsed:.1f}秒")
    print(f"嵌入维度: {embeddings.shape}")
    
    # 保存缓存
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    cache_data = {
        'embeddings': embeddings.tolist(),
        'names': df['name'].tolist(),
        'cn_names': df['cn_name'].tolist(),
        'post_counts': df['post_count'].tolist() if 'post_count' in df.columns else ['0'] * len(df),
        'categories': df['category'].tolist() if 'category' in df.columns else ['0'] * len(df),
        'nsfw': df['nsfw'].tolist() if 'nsfw' in df.columns else ['0'] * len(df),
    }
    
    with open(EMBED_CACHE_PATH, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"嵌入缓存已保存到: {EMBED_CACHE_PATH}")
    
    # 删除检查点文件
    try:
        os.remove(CHECKPOINT_PATH)
    except:
        pass
    
    return embeddings, df


def build_reranker_cache():
    """构建重排缓存（元数据）"""
    print("\n" + "="*50)
    print("步骤2: 准备重排模型...")
    print("="*50)
    
    # 重新加载CSV获取所有文本
    df = load_csv()
    
    # 准备文本
    texts = []
    for _, row in df.iterrows():
        name = row.get('name', '')
        cn_name = row.get('cn_name', '')
        
        if cn_name:
            text = f"{name} {cn_name.replace(',', ' ')}"
        else:
            text = name
        
        texts.append(text)
    
    rerank_cache = {
        'names': df['name'].tolist(),
        'cn_names': df['cn_name'].tolist(),
        'texts': texts,
        'post_counts': df['post_count'].tolist() if 'post_count' in df.columns else ['0'] * len(df),
        'categories': df['category'].tolist() if 'category' in df.columns else ['0'] * len(df),
        'nsfw': df['nsfw'].tolist() if 'nsfw' in df.columns else ['0'] * len(df),
    }
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    with open(RERANK_CACHE_PATH, 'wb') as f:
        pickle.dump(rerank_cache, f)
    
    print(f"重排缓存已保存到: {RERANK_CACHE_PATH}")
    print(f"重排模型: {RERANKER_MODEL}")
    
    return rerank_cache


def update_semantic_search():
    """更新semantic_search.py以使用新缓存"""
    print("\n" + "="*50)
    print("步骤3: 验证缓存...")
    print("="*50)
    
    # 验证嵌入缓存
    if os.path.exists(EMBED_CACHE_PATH):
        with open(EMBED_CACHE_PATH, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ 嵌入缓存: {len(data['names'])} 个标签")
    else:
        print("✗ 嵌入缓存不存在!")
        return False
    
    # 验证重排缓存
    if os.path.exists(RERANK_CACHE_PATH):
        with open(RERANK_CACHE_PATH, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ 重排缓存: {len(data['names'])} 个标签")
    else:
        print("✗ 重排缓存不存在!")
        return False
    
    print("\n缓存构建完成!")
    return True


def main():
    """主函数"""
    print("="*60)
    print("Danbooru Tag Embedding Cache Builder")
    print("  - 嵌入模型: Pro/BAAI/bge-m3")
    print("  - 重排模型: Pro/BAAI/bge-reranker-v2-m3")
    print("="*60)
    
    # 步骤1: 加载CSV
    df = load_csv()
    
    # 步骤2: 构建嵌入缓存
    build_embeddings_cache(df)
    
    # 步骤3: 构建重排缓存
    build_reranker_cache()
    
    # 步骤4: 验证
    update_semantic_search()
    
    print("\n" + "="*60)
    print("全部完成!")
    print("="*60)


if __name__ == '__main__':
    main()