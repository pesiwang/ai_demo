from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

openai_llm = OpenAI(temperature=0.9)

#text = "What would be a good company name for a company that makes colorful socks?"
#print(llm(text))

prompt_tpl = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

prompt_str = prompt_tpl.format(product="colorful socks")
print(prompt_str)
#print(llm(prompt_str))

chain = LLMChain(llm=openai_llm, prompt=prompt_tpl)
result = chain.run("colorful socks")
print(result)


