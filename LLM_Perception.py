"""
Project: Urban Leisure Space Perception Analysis via LLM
Script Name: Reliability Consistency Test (模型推理稳定性检验脚本)
Author: [Your Name/Team]
Date: 2025-10

Description:
    本脚本旨在验证大语言模型（Qwen-Turbo）在处理非结构化文本分析任务时的推理稳定性与可复核性。
    实验设计：
    1. 从源数据库中随机抽取 500 条样本数据。
    2. 在固定温度参数（temperature=0.1）和随机种子（seed=888）下，对同一批数据进行 3 次独立重复分析。
    3. 计算并输出三次运行的一致性指标，包括：
       - 整体情感一致率 (Overall Sentiment Consistency)
       - 维度识别一致率 (Dimension Recognition Consistency)
       - Cohen's Kappa 系数

Note:
    本代码的核心分析逻辑与正式研究完全一致。
    若需进行全量数据的正式分析，只需将本脚本中的"重复运行3次"逻辑移除，并将数据源改为全量遍历即可。
"""

import sqlite3
import json
import time
import random
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np

# ==================== 配置区 ====================
API_KEY = "你的APIKEY"  # 请替换为实际的 DashScope API Key
DB_PATH = r"你的数据库路径"  # 请替换为实际的 SQLite 数据库路径
OUTPUT_FOLDER = "reliability_test_results"

# 实验参数设置
SAMPLE_SIZE = 500  # 抽取样本数量 (由200调整为500以增强统计效力)
NUM_RUNS = 3       # 重复运行次数
SEED = 888         # 固定随机种子以确保抽取样本的可复现性

# OpenAI客户端配置 (适配阿里云 DashScope)
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 创建输出文件夹
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# ==================== 分析提示词 (Prompt Engineering) ====================
# 该提示词采用思维链 (Chain-of-Thought) 和少样本学习 (Few-Shot) 策略
ANALYSIS_PROMPT = """
请分析以下关于城市休闲空间的中文文本：

{text}

**重要说明**：
1. 首先将文本分解为独立的句子
2. **对每个句子分别独立分析**，判断该句子涉及的维度和情感倾向
3. 不同句子可能涉及完全不同的维度，请分别判断，不要将整个文本的维度都分配给每个句子
4. 提取文本中提及的具体地点

**分析维度**：

**感知可达性 (perceived_accessibility)**
定义：公众对于到达和进入休闲场所的便捷性与友好性的主观感受
包括：交通便利性、寻路导航、距离感知、入口设计、无障碍设施、营业时间等
示例表达："地铁直达"、"藏在巷子里，很难找"、"公交很方便"、"停车位不够"

**功能多样性 (functional_diversity)**
定义：空间所提供的服务和设施的多样性和完整性
包括：活动种类、设施完备性、多功能使用、适应不同用户需求等
示例表达："不光能看展，还能滑板、跑步、遛狗"、"除了购物没别的了"、"功能很单一"

**视觉景观 (spatial_design_visual_aesthetics)**
定义：对空间整体布局、建筑风格、景观设计等视觉元素的艺术性与美感的评价
包括：布局设计、建筑风格、装饰、色彩搭配、绿化、清洁度、照明等
示例表达："植物搭配很有层次"、"铺装单调，毫无特色"、"设计很有格调"、"破破烂烂的"

**文化艺术 (cultural_artistic_atmosphere)**
定义：空间所承载或营造出的历史底蕴、艺术气息与特定文化主题的整体氛围
包括：文化遗产、艺术展示、历史联系、地方特色、文化活动等
示例表达："老洋房很有海派风情"、"搞了些假古董就说是复古"、"很有历史感"、"文化氛围浓厚"

**社交互动 (social_interaction)**
定义：空间作为促进人与人之间交流、互动与情感连接的媒介作用和潜力
包括：聚会空间、社会联系、团体活动、社区活动、包容性等
示例表达："很适合情侣约会"、"在这里认识了很多有趣的朋友"、"人很多很热闹"、"太冷清了"

**商业消费 (commercial_vitality_consumption)**
定义：对空间内商业业态的丰富度、独特性、服务质量及整体消费过程的综合评价
包括：零售多样性、餐饮选择、服务质量、价格水平、消费体验等
示例表达："有很多好逛的买手店"、"网红餐厅，排队两小时"、"太商业化了"、"东西很贵"

**环境心理 (leisure_experience_comfort)**
定义：个体在空间中感受到的身心放松、压力疏解、安全、舒适的综合体验
包括：座椅设施、噪音水平、温度舒适度、人群密度、安全感、放松感等
示例表达："滨江步道吹风，整个人都放松了"、"缺少遮阳的地方"、"太吵了"、"很舒服"

**情感倾向判断**：
- positive（正面）：表达赞美、喜爱、满意等积极情感
- negative（负面）：表达批评、不满、失望等消极情感  
- neutral（中性）：客观描述，无明显情感倾向

**请务必记住**：每个句子要根据其具体内容独立分析维度，不要所有句子都分配相同的维度。

请仅返回以下格式的JSON对象：
{{
  "places": ["地点1", "地点2"] 或 "未提及具体地点",
  "overall_sentiment": "positive/negative/neutral",
  "sentence_analysis": [
    {{
      "sentence": "句子1的实际内容",
      "dimensions": ["该句子涉及的具体维度"],
      "sentiment": "该句子的情感倾向"
    }},
    {{
      "sentence": "句子2的实际内容", 
      "dimensions": ["该句子涉及的具体维度"],
      "sentiment": "该句子的情感倾向"
    }}
  ]
}}
"""

# ==================== 辅助函数 ====================
def extract_json_from_text(text):
    """从大模型返回的文本中健壮地提取JSON对象"""
    import re
    text = text.strip()
    
    try:
        return json.loads(text)
    except:
        pass
    
    # 清洗 Markdown 标记
    text = re.sub(r'^```(?:json)?', '', text)
    text = re.sub(r'```$', '', text)
    text = text.strip()
    
    try:
        return json.loads(text)
    except:
        pass
    
    # 尝试正则提取 JSON 部分
    match = re.search(r'({.*})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # 返回默认空值结构
    return {
        "places": "未提及具体地点",
        "overall_sentiment": "neutral",
        "sentence_analysis": []
    }

def analyze_text_with_params(text, temperature=0.1, top_p=0.95, seed=888):
    """
    调用 Qwen 模型进行文本分析
    
    Args:
        temperature (float): 控制随机性，0.1 接近贪婪解码，保证稳定性。
        seed (int): 固定随机种子，确保可复现性。
    """
    if not isinstance(text, str) or text.strip() == "":
        return {
            "places": "未提及具体地点",
            "overall_sentiment": "neutral",
            "sentence_analysis": []
        }
    
    try:
        messages = [
            {"role": "system", "content": "你是一个专业的中文文本分析助手，擅长按句子级别分析城市空间评价。请务必对每个句子独立分析维度，不要给所有句子分配相同的维度。只返回有效的JSON格式，不包含其他文字。"},
            {"role": "user", "content": ANALYSIS_PROMPT.format(text=text)}
        ]
        
        completion = client.chat.completions.create(
            model="qwen-turbo-latest",  # 注意：对应论文研究期间（2025-07-15）的 Qwen-Turbo 版本
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_tokens=4000,
        )
        
        response = completion.choices[0].message.content
        result = extract_json_from_text(response)
        
        if result and 'sentence_analysis' in result:
            return result
        else:
            return {
                "places": "未提及具体地点",
                "overall_sentiment": "neutral",
                "sentence_analysis": []
            }
    
    except Exception as e:
        print(f"API调用错误: {str(e)}")
        return {
            "places": "未提及具体地点",
            "overall_sentiment": "neutral",
            "sentence_analysis": []
        }

# ==================== 主要功能函数 ====================
def sample_data_from_db(db_path, sample_size=500, random_seed=888):
    """从 SQLite 数据库中随机抽取指定数量的样本"""
    print(f"正在从数据库抽取 {sample_size} 条样本...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    if not tables:
        raise ValueError("数据库中没有找到表")
    table_name = tables[0][0]
    
    # 获取列信息
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    
    # 智能识别主键列
    pk_column = None
    for col in columns:
        if col[5] == 1:
            pk_column = col[1]
            break
    if not pk_column:
        pk_column = "rowid"
    
    # 智能识别内容列（查找包含 '内容'/'content'/'text' 的列，或最长的文本列）
    content_column = None
    for col in column_names:
        if '内容' in col or 'content' in col or 'text' in col:
            content_column = col
            break
    
    if not content_column:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
        row = cursor.fetchone()
        for i, col in enumerate(column_names):
            if row[i] and isinstance(row[i], str) and len(row[i]) > 100:
                content_column = col
                break
    
    print(f"表名: {table_name}, 主键列: {pk_column}, 内容列: {content_column}")
    
    # 获取总行数
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = cursor.fetchone()[0]
    print(f"数据库总行数: {total_rows}")
    
    # 随机抽取
    random.seed(random_seed)
    cursor.execute(f"SELECT {pk_column}, {content_column} FROM {table_name}")
    all_data = cursor.fetchall()
    
    # 确保样本量不超过总数据量
    actual_sample_size = min(sample_size, len(all_data))
    sampled_data = random.sample(all_data, actual_sample_size)
    
    conn.close()
    
    print(f"成功抽取 {len(sampled_data)} 条样本")
    return sampled_data

def run_analysis(samples, run_number):
    """对样本集运行一次完整分析"""
    print(f"\n{'='*50}")
    print(f"开始第 {run_number} 次分析 (Run {run_number}/{NUM_RUNS})")
    print(f"{'='*50}")
    
    results = []
    
    for idx, (sample_id, content) in enumerate(tqdm(samples, desc=f"Run {run_number}")):
        try:
            # 使用预设的稳定参数进行分析
            result = analyze_text_with_params(
                text=content,
                temperature=0.1,
                top_p=0.95,
                seed=SEED
            )
            
            results.append({
                'sample_id': sample_id,
                'content': content,
                'overall_sentiment': result.get('overall_sentiment', 'neutral'),
                'sentence_analysis': result.get('sentence_analysis', [])
            })
            
            # 简单的限流保护 (每10条暂停1秒)
            if (idx + 1) % 10 == 0:
                time.sleep(1)
        
        except Exception as e:
            print(f"处理样本 {sample_id} 时出错: {e}")
            results.append({
                'sample_id': sample_id,
                'content': content,
                'overall_sentiment': 'neutral',
                'sentence_analysis': []
            })
    
    return results

def extract_dimensions_and_sentiments(results):
    """辅助函数：提取结果中的维度和情感标签用于比较"""
    dimensions_list = []
    sentiments_list = []
    
    for result in results:
        # 整体情感
        overall_sentiment = result.get('overall_sentiment', 'neutral')
        sentiments_list.append(overall_sentiment)
        
        # 提取所有句子涉及的维度（去重）
        all_dimensions = set()
        sentence_analysis = result.get('sentence_analysis', [])
        
        for sentence in sentence_analysis:
            dims = sentence.get('dimensions', [])
            if isinstance(dims, list):
                all_dimensions.update(dims)
        
        # 转换为排序后的字符串（便于比较是否完全一致）
        dimensions_str = ','.join(sorted(all_dimensions)) if all_dimensions else 'none'
        dimensions_list.append(dimensions_str)
    
    return dimensions_list, sentiments_list

def calculate_consistency(run1_results, run2_results, run3_results):
    """计算三次运行的一致性指标 (Consistency Metrics)"""
    print(f"\n{'='*50}")
    print("计算一致性指标 (Consistency Metrics Calculation)")
    print(f"{'='*50}")
    
    # 提取数据
    dims1, sents1 = extract_dimensions_and_sentiments(run1_results)
    dims2, sents2 = extract_dimensions_and_sentiments(run2_results)
    dims3, sents3 = extract_dimensions_and_sentiments(run3_results)
    
    total = len(sents1)
    
    # 1. 整体情感的完全一致率 (Overall Sentiment Agreement)
    sentiment_match_12 = sum(1 for a, b in zip(sents1, sents2) if a == b)
    sentiment_match_13 = sum(1 for a, b in zip(sents1, sents3) if a == b)
    sentiment_match_23 = sum(1 for a, b in zip(sents2, sents3) if a == b)
    
    print(f"\n【整体情感一致性 (Sentiment Consistency)】")
    print(f"Run1 vs Run2: {sentiment_match_12}/{total} = {sentiment_match_12/total*100:.2f}%")
    print(f"Run1 vs Run3: {sentiment_match_13}/{total} = {sentiment_match_13/total*100:.2f}%")
    print(f"Run2 vs Run3: {sentiment_match_23}/{total} = {sentiment_match_23/total*100:.2f}%")
    print(f"平均一致率: {(sentiment_match_12+sentiment_match_13+sentiment_match_23)/(3*total)*100:.2f}%")
    
    # 2. 维度的完全一致率 (Dimension Agreement)
    dims_match_12 = sum(1 for a, b in zip(dims1, dims2) if a == b)
    dims_match_13 = sum(1 for a, b in zip(dims1, dims3) if a == b)
    dims_match_23 = sum(1 for a, b in zip(dims2, dims3) if a == b)
    
    print(f"\n【维度编码一致性 (Dimension Consistency)】")
    print(f"Run1 vs Run2: {dims_match_12}/{total} = {dims_match_12/total*100:.2f}%")
    print(f"Run1 vs Run3: {dims_match_13}/{total} = {dims_match_13/total*100:.2f}%")
    print(f"Run2 vs Run3: {dims_match_23}/{total} = {dims_match_23/total*100:.2f}%")
    print(f"平均一致率: {(dims_match_12+dims_match_13+dims_match_23)/(3*total)*100:.2f}%")
    
    # 3. Cohen's Kappa（情感）
    try:
        kappa_12_sent = cohen_kappa_score(sents1, sents2)
        kappa_13_sent = cohen_kappa_score(sents1, sents3)
        kappa_23_sent = cohen_kappa_score(sents2, sents3)
        avg_kappa_sent = (kappa_12_sent + kappa_13_sent + kappa_23_sent) / 3
        
        print(f"\n【Cohen's Kappa系数 (Sentiment)】")
        print(f"Run1 vs Run2: {kappa_12_sent:.4f}")
        print(f"Run1 vs Run3: {kappa_13_sent:.4f}")
        print(f"Run2 vs Run3: {kappa_23_sent:.4f}")
        print(f"平均Kappa: {avg_kappa_sent:.4f}")
        
        if avg_kappa_sent > 0.8:
            print("解释: 几乎完美一致 (Almost Perfect Agreement)")
        elif avg_kappa_sent > 0.6:
            print("解释: 高度一致 (Substantial Agreement)")
        elif avg_kappa_sent > 0.4:
            print("解释: 中度一致 (Moderate Agreement)")
        else:
            print("解释: 一致性较低 (Fair/Slight Agreement)")
    
    except Exception as e:
        print(f"\n计算Kappa系数时出错: {e}")
        avg_kappa_sent = None
    
    # 4. 三次完全一致 (Perfect Match across all 3 runs)
    perfect_match_sentiment = sum(1 for a, b, c in zip(sents1, sents2, sents3) if a == b == c)
    perfect_match_dimensions = sum(1 for a, b, c in zip(dims1, dims2, dims3) if a == b == c)
    
    print(f"\n【三次完全一致 (Perfect Match)】")
    print(f"情感三次完全一致: {perfect_match_sentiment}/{total} = {perfect_match_sentiment/total*100:.2f}%")
    print(f"维度三次完全一致: {perfect_match_dimensions}/{total} = {perfect_match_dimensions/total*100:.2f}%")
    
    # 返回统计字典
    stats = {
        'sample_size': total,
        'sentiment_consistency': {
            'run1_vs_run2': sentiment_match_12 / total,
            'run1_vs_run3': sentiment_match_13 / total,
            'run2_vs_run3': sentiment_match_23 / total,
            'average': (sentiment_match_12 + sentiment_match_13 + sentiment_match_23) / (3 * total),
            'perfect_match_all_three': perfect_match_sentiment / total
        },
        'dimension_consistency': {
            'run1_vs_run2': dims_match_12 / total,
            'run1_vs_run3': dims_match_13 / total,
            'run2_vs_run3': dims_match_23 / total,
            'average': (dims_match_12 + dims_match_13 + dims_match_23) / (3 * total),
            'perfect_match_all_three': perfect_match_dimensions / total
        },
        'cohens_kappa_sentiment': {
            'average': avg_kappa_sent
        }
    }
    
    return stats

def save_results(samples, run1, run2, run3, stats):
    """保存结果到文件"""
    # 1. 保存详细对比结果 (JSON)
    detailed_results = []
    for i, (sample_id, content) in enumerate(samples):
        detailed_results.append({
            'sample_id': sample_id,
            'content': content,
            'run1_sentiment': run1[i]['overall_sentiment'],
            'run2_sentiment': run2[i]['overall_sentiment'],
            'run3_sentiment': run3[i]['overall_sentiment'],
            'run1_analysis': run1[i]['sentence_analysis'],
            'run2_analysis': run2[i]['sentence_analysis'],
            'run3_analysis': run3[i]['sentence_analysis']
        })
    
    with open(f"{OUTPUT_FOLDER}/detailed_results.json", 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    # 2. 保存统计指标 (JSON)
    with open(f"{OUTPUT_FOLDER}/consistency_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 3. 保存Excel对比表 (方便人工查看)
    df_data = []
    for i, (sample_id, content) in enumerate(samples):
        dims1, _ = extract_dimensions_and_sentiments([run1[i]])
        dims2, _ = extract_dimensions_and_sentiments([run2[i]])
        dims3, _ = extract_dimensions_and_sentiments([run3[i]])
        
        df_data.append({
            'ID': sample_id,
            '内容摘要': content[:50] + '...' if len(content) > 50 else content,
            'Run1_情感': run1[i]['overall_sentiment'],
            'Run2_情感': run2[i]['overall_sentiment'],
            'Run3_情感': run3[i]['overall_sentiment'],
            '情感一致': '✓' if run1[i]['overall_sentiment'] == run2[i]['overall_sentiment'] == run3[i]['overall_sentiment'] else '✗',
            'Run1_维度': dims1[0],
            'Run2_维度': dims2[0],
            'Run3_维度': dims3[0],
            '维度一致': '✓' if dims1[0] == dims2[0] == dims3[0] else '✗'
        })
    
    df = pd.DataFrame(df_data)
    df.to_excel(f"{OUTPUT_FOLDER}/reliability_comparison.xlsx", index=False)
    
    print(f"\n结果已保存到 {OUTPUT_FOLDER} 文件夹")
    print(f"- detailed_results.json (详细分析结果)")
    print(f"- consistency_stats.json (一致性统计指标)")
    print(f"- reliability_comparison.xlsx (Excel对比表)")

# ==================== 主函数 ====================
def main():
    print("="*60)
    print("大语言模型分析稳定性验证程序 (LLM Analysis Reliability Test)")
    print("="*60)
    print(f"配置信息:")
    print(f"- 样本数量: {SAMPLE_SIZE} (Random Sample)")
    print(f"- 运行次数: {NUM_RUNS} (Repetitions)")
    print(f"- 模型参数: Temperature=0.1, Seed={SEED}")
    print("="*60)
    
    # 1. 抽取样本
    try:
        samples = sample_data_from_db(DB_PATH, SAMPLE_SIZE, SEED)
    except Exception as e:
        print(f"读取数据库失败: {e}")
        return
    
    # 2. 运行三次分析
    run1_results = run_analysis(samples, 1)
    time.sleep(2)  # 避免API限流缓冲
    
    run2_results = run_analysis(samples, 2)
    time.sleep(2)
    
    run3_results = run_analysis(samples, 3)
    
    # 3. 计算一致性
    stats = calculate_consistency(run1_results, run2_results, run3_results)
    
    # 4. 保存结果
    save_results(samples, run1_results, run2_results, run3_results, stats)
    
    print("\n" + "="*60)
    print("验证完成！(Verification Completed)")
    print("="*60)

if __name__ == "__main__":
    main()