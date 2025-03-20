from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
import pandas as pd

def read_table(file_path='document.csv'):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"读取文件失败：{e}")
        return None

def translate_to_english(hanja):
    prompt_template = PromptTemplate.from_template("请将'{hanja}'翻译成一个简单的英语单词（不超过6个字母，不重复）。")
    ollama_llm = Ollama(model="qwen2:latest")
    llm_chain = LLMChain(llm=ollama_llm, prompt=prompt_template)
    prompt = prompt_template.format(hanja=hanja)
    result = llm_chain.invoke(prompt)
    result_text = result.get('text', '').strip()
    return result_text

def save_to_csv(data, output_file='fanyi.csv'):
    data.to_csv(output_file, index=False, encoding='utf-8-sig')

def main():
    file_path = 'document.csv'
    data = read_table(file_path)
    if data is None:
        return
    
    output_file = 'fanyi.csv'
    unique_words = set()
    for index, row in data.iterrows():
        hanja = row['汉语']
        translated_word = None
        for attempt in range(3):
            current_translation = translate_to_english(hanja)
            if current_translation not in unique_words:
                translated_word = current_translation
                break
        
        if translated_word and translated_word not in unique_words:
            unique_words.add(translated_word)
            data.loc[index, '英语'] = translated_word
        else:
            data.loc[index, '英语'] = "未翻译"
        save_to_csv(data, output_file)
        print(f"已处理汉字：{hanja}，翻译结果：{data.loc[index, '英语']}")
    
    print("翻译完成，已保存到fanyi.csv")

if __name__ == "__main__":
    main()
