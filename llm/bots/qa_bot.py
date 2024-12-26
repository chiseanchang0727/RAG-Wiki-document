from .base import LLMBase
from llm.config.llm_config import LLMConfig
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.chat import ChatPromptTemplate

class QABot(LLMBase):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.model = self.get_llm()
       
    def get_answer(self, prompt, target, question, reference):
        
        prompt = ChatPromptTemplate.from_template(prompt)
        
        chain = (
            {
                "target": RunnablePassthrough(),
                "question": RunnablePassthrough(), 
                "reference": RunnablePassthrough()
            }
            | prompt
            | self.model
            | StrOutputParser()
        )
        
        return chain.invoke({"target":target, "question": question, "reference": reference})
    
