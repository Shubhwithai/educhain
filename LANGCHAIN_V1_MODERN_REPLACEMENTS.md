# LangChain v1.0 Modern Replacements Guide
## Migrate from langchain-classic to Future-Proof Code

---

## 🚨 Why Replace langchain-classic?

`langchain-classic` is a **temporary compatibility package** containing deprecated code. It will be removed in future versions. This guide shows you how to replace legacy patterns with modern LangChain v1.0 approaches.

---

## 📋 Current Educhain Usage of Legacy Code

### ❌ **Legacy Code in Educhain**

**File:** `educhain/engines/qna_engine.py`

```python
# LEGACY - Will be deprecated
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

def _setup_retrieval_qa(self, vector_store: Chroma) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=self.llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )
```

**Used in:** `generate_questions_with_rag()` method

---

## ✅ Modern LangChain v1.0 Replacements

LangChain v1.0 offers **TWO modern approaches** to replace `RetrievalQA`:

### **Option 1: RAG Agent (Recommended for Complex Queries)** 🤖
- Uses tool-based retrieval
- LLM decides when to search
- Flexible and powerful
- Best for: Multi-step reasoning, complex queries

### **Option 2: RAG Chain with Middleware (Recommended for Simple Queries)** ⚡
- Direct context injection
- Single LLM call per query
- Fast and efficient
- Best for: Straightforward Q&A, performance-critical applications

---

## 🔧 Complete Migration Examples

### **Option 1: RAG Agent Pattern (Recommended)**

This is the **most flexible and future-proof** approach using LangChain v1.0's agent system.

#### **New Imports (Add to qna_engine.py):**
```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
```

#### **New Implementation:**

```python
# MODERN APPROACH - LangChain v1.0
class QnAEngine:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        if llm_config is None:
            llm_config = LLMConfig()
        self.llm = self._initialize_llm(llm_config)
        self.embeddings = None
        self.vector_store = None  # Store vector store for retrieval

    def _create_retrieval_tool(self, vector_store):
        """Create a retrieval tool for the agent"""
        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str) -> str:
            """Retrieve relevant context from the knowledge base to answer questions."""
            # Retrieve top k relevant documents
            retrieved_docs = vector_store.similarity_search(query, k=4)
            
            # Format the retrieved content
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        
        return retrieve_context

    def _setup_rag_agent(self, vector_store):
        """Setup RAG agent using modern LangChain v1.0 pattern"""
        # Create retrieval tool
        retrieve_tool = self._create_retrieval_tool(vector_store)
        
        # Define system prompt for question generation
        system_prompt = """
        You are an expert educational content generator specialized in creating high-quality questions.
        
        You have access to a retrieval tool that can fetch relevant content from the knowledge base.
        Use this tool to gather context before generating questions.
        
        When generating questions:
        1. First, use the retrieve_context tool to get relevant information
        2. Analyze the retrieved content thoroughly
        3. Generate questions that are clear, accurate, and pedagogically sound
        4. Ensure questions align with the specified learning objectives and difficulty level
        """
        
        # Create agent with retrieval tool
        agent = create_agent(
            model=self.llm,
            tools=[retrieve_tool],
            system_prompt=system_prompt
        )
        
        return agent

    def generate_questions_with_rag(
        self,
        source: str,
        source_type: str,
        num: int,
        question_type: QuestionType = "Multiple Choice",
        prompt_template: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        response_model: Optional[Type[Any]] = None,
        learning_objective: Optional[str] = None,
        difficulty_level: Optional[str] = None,
        output_format: Optional[OutputFormatType] = None,
        **kwargs
    ) -> Any:
        """
        Generate questions using modern RAG agent pattern (LangChain v1.0)
        """
        if self.embeddings is None:
            self.embeddings = OpenAIEmbeddings()

        # Load and process content
        content = self._load_data(source, source_type)
        vector_store = self._create_vector_store(content)
        
        # Setup RAG agent (modern approach)
        rag_agent = self._setup_rag_agent(vector_store)
        
        # Get parser and model for structured output
        parser, model = self._get_parser_and_model(question_type, response_model)
        format_instructions = parser.get_format_instructions()
        
        # Build the query for the agent
        query = f"""
        Generate {num} {question_type} questions based on the content in the knowledge base.
        
        Requirements:
        - Question Type: {question_type}
        - Number of Questions: {num}
        """
        
        if learning_objective:
            query += f"\n- Learning Objective: {learning_objective}"
        
        if difficulty_level:
            query += f"\n- Difficulty Level: {difficulty_level}"
        
        if custom_instructions:
            query += f"\n- Additional Instructions: {custom_instructions}"
        
        query += f"""
        
        Output Format:
        {format_instructions}
        
        Important: First retrieve relevant content from the knowledge base, 
        then generate questions based on that content.
        """
        
        # Invoke the agent
        response = rag_agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        
        # Extract the generated content
        result_content = response["messages"][-1].content
        
        try:
            # Parse the structured output
            structured_output = parser.parse(result_content)
            
            if output_format:
                self._handle_output_format(structured_output, output_format)
            
            return structured_output
        except Exception as e:
            print(f"Error parsing output: {e}")
            print("Raw output:", result_content)
            return model()
```

#### **Benefits of RAG Agent:**
- ✅ **Future-proof**: Uses LangChain v1.0's agent system
- ✅ **Flexible**: Agent decides when to retrieve
- ✅ **Multi-step reasoning**: Can retrieve multiple times if needed
- ✅ **Tool-based**: Easy to add more tools (web search, calculator, etc.)
- ✅ **No deprecated code**: Zero dependency on `langchain-classic`

---

### **Option 2: RAG Chain with Middleware (Faster Alternative)**

This approach uses middleware to inject context before the model call - **single LLM call, very fast**.

#### **New Imports (Add to qna_engine.py):**
```python
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, dynamic_prompt, ModelRequest
from langchain_core.documents import Document
from typing import Any
```

#### **Implementation:**

```python
# MODERN APPROACH - Fast RAG Chain with Middleware
class QnAEngine:
    
    def _setup_rag_chain_with_middleware(self, vector_store):
        """Setup RAG chain using middleware for context injection"""
        
        # Approach 2A: Simple Dynamic Prompt
        @dynamic_prompt
        def prompt_with_context(request: ModelRequest) -> str:
            """Inject retrieved context into the prompt"""
            last_query = request.state["messages"][-1].text
            
            # Retrieve relevant documents
            retrieved_docs = vector_store.similarity_search(last_query, k=4)
            docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            # Create system message with context
            system_message = (
                "You are an expert educational content generator.\n\n"
                "Use the following context from the knowledge base to generate questions:\n\n"
                f"{docs_content}\n\n"
                "Generate high-quality educational questions based on this context."
            )
            return system_message
        
        # Create agent with middleware
        agent = create_agent(
            model=self.llm,
            tools=[],  # No tools needed - direct context injection
            middleware=[prompt_with_context]
        )
        
        return agent

    def _setup_rag_chain_with_state(self, vector_store):
        """Setup RAG chain with custom state to store retrieved documents"""
        
        # Define custom state to store context
        class State(AgentState):
            context: list[Document]
        
        # Custom middleware to retrieve and inject context
        class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
            state_schema = State
            
            def __init__(self, vector_store):
                self.vector_store = vector_store
            
            def before_model(self, state: AgentState) -> dict[str, Any] | None:
                """Retrieve documents before model call"""
                last_message = state["messages"][-1]
                
                # Retrieve relevant documents
                retrieved_docs = self.vector_store.similarity_search(
                    last_message.text, k=4
                )
                
                # Format context
                docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
                
                # Augment the message with context
                augmented_message_content = (
                    f"{last_message.text}\n\n"
                    "Use the following context from the knowledge base:\n"
                    f"{docs_content}"
                )
                
                return {
                    "messages": [
                        last_message.model_copy(
                            update={"content": augmented_message_content}
                        )
                    ],
                    "context": retrieved_docs,
                }
        
        # Create agent with custom middleware
        agent = create_agent(
            model=self.llm,
            tools=[],
            middleware=[RetrieveDocumentsMiddleware(vector_store)]
        )
        
        return agent

    def generate_questions_with_rag_fast(
        self,
        source: str,
        source_type: str,
        num: int,
        question_type: QuestionType = "Multiple Choice",
        prompt_template: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        response_model: Optional[Type[Any]] = None,
        learning_objective: Optional[str] = None,
        difficulty_level: Optional[str] = None,
        output_format: Optional[OutputFormatType] = None,
        **kwargs
    ) -> Any:
        """
        Fast RAG chain using middleware (single LLM call)
        """
        if self.embeddings is None:
            self.embeddings = OpenAIEmbeddings()

        # Load and process content
        content = self._load_data(source, source_type)
        vector_store = self._create_vector_store(content)
        
        # Setup RAG chain with middleware
        rag_chain = self._setup_rag_chain_with_middleware(vector_store)
        # OR use: rag_chain = self._setup_rag_chain_with_state(vector_store)
        
        # Get parser and model
        parser, model = self._get_parser_and_model(question_type, response_model)
        format_instructions = parser.get_format_instructions()
        
        # Build query
        query = f"""
        Generate {num} {question_type} questions.
        
        Requirements:
        - Question Type: {question_type}
        - Number of Questions: {num}
        """
        
        if learning_objective:
            query += f"\n- Learning Objective: {learning_objective}"
        
        if difficulty_level:
            query += f"\n- Difficulty Level: {difficulty_level}"
        
        if custom_instructions:
            query += f"\n- Additional Instructions: {custom_instructions}"
        
        query += f"\n\nOutput Format:\n{format_instructions}"
        
        # Invoke the chain (context is automatically injected)
        response = rag_chain.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        
        # Extract result
        result_content = response["messages"][-1].content
        
        try:
            structured_output = parser.parse(result_content)
            
            if output_format:
                self._handle_output_format(structured_output, output_format)
            
            return structured_output
        except Exception as e:
            print(f"Error parsing output: {e}")
            return model()
```

#### **Benefits of RAG Chain with Middleware:**
- ⚡ **Fast**: Single LLM call per query
- ✅ **Simple**: Straightforward implementation
- ✅ **Future-proof**: No deprecated code
- ✅ **Efficient**: Lower latency and cost
- ✅ **State management**: Can store retrieved docs in state

---

## 🔄 Migration Path: Step-by-Step

### **Phase 1: Update Imports**

**Remove:**
```python
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
```

**Add:**
```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import AgentMiddleware, AgentState, dynamic_prompt, ModelRequest
from langchain_core.documents import Document
from typing import Any
```

### **Phase 2: Replace _setup_retrieval_qa Method**

**Delete Legacy Method:**
```python
def _setup_retrieval_qa(self, vector_store: Chroma) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=self.llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )
```

**Add Modern Methods:**
- Add `_create_retrieval_tool()` for agent approach
- Add `_setup_rag_agent()` for agent approach
- OR add `_setup_rag_chain_with_middleware()` for chain approach

### **Phase 3: Update generate_questions_with_rag Method**

**Replace line:**
```python
qa_chain = self._setup_retrieval_qa(vector_store)
```

**With:**
```python
rag_agent = self._setup_rag_agent(vector_store)
# OR for faster approach:
# rag_chain = self._setup_rag_chain_with_middleware(vector_store)
```

### **Phase 4: Update setup.py**

**Remove:**
```python
"langchain-classic",  # Will be deprecated
```

**Ensure these are present:**
```python
"langchain>=1.0.0",
"langchain-core>=1.0.0",
"langchain-openai>=0.2.0",
"langchain-community>=0.3.0",
```

### **Phase 5: Test Thoroughly**

- [ ] Test RAG-based question generation with PDFs
- [ ] Test RAG-based question generation with URLs
- [ ] Test RAG-based question generation with text
- [ ] Test with different question types (MCQ, Short Answer, etc.)
- [ ] Test with learning objectives and difficulty levels
- [ ] Compare output quality with legacy RetrievalQA

---

## 📊 Comparison: Legacy vs Modern

| Feature | Legacy RetrievalQA | RAG Agent (v1.0) | RAG Chain (v1.0) |
|---------|-------------------|------------------|------------------|
| **Status** | ❌ Deprecated | ✅ Modern | ✅ Modern |
| **Package** | langchain-classic | langchain | langchain |
| **Future-proof** | ❌ No | ✅ Yes | ✅ Yes |
| **Flexibility** | ⚠️ Limited | ✅ High | ⚠️ Medium |
| **Speed** | ⚠️ Medium | ⚠️ Multiple calls | ✅ Single call |
| **Multi-step reasoning** | ❌ No | ✅ Yes | ❌ No |
| **Tool extensibility** | ❌ No | ✅ Yes | ❌ No |
| **State management** | ⚠️ Limited | ✅ Advanced | ✅ Good |
| **Best for** | N/A | Complex queries | Simple Q&A |

---

## 🎯 Recommended Approach for Educhain

### **Primary Recommendation: RAG Agent** 🏆

Use the **RAG Agent pattern** because:

1. **Future-proof**: Aligned with LangChain v1.0 direction
2. **Flexible**: Can handle complex question generation scenarios
3. **Extensible**: Easy to add more tools (web search, calculators, etc.)
4. **Intelligent**: Agent decides when to retrieve more context
5. **Better quality**: Multi-step reasoning improves question quality

### **Alternative: RAG Chain for Performance**

Use **RAG Chain with Middleware** if:
- Performance is critical (single LLM call)
- Questions are straightforward (no multi-step reasoning needed)
- Lower latency is required
- Cost optimization is important

---

## 💡 Additional Modern Patterns

### **1. Hybrid RAG (Advanced)**

Combines retrieval with self-correction and validation:

```python
from langchain.agents import create_agent

def setup_hybrid_rag(vector_store, llm):
    """RAG with query enhancement and validation"""
    
    @tool
    def retrieve_with_enhancement(query: str) -> str:
        """Retrieve with automatic query enhancement"""
        # Could rewrite query, generate variations, etc.
        enhanced_query = enhance_query(query)
        docs = vector_store.similarity_search(enhanced_query, k=5)
        return format_docs(docs)
    
    @tool
    def validate_retrieval(query: str, context: str) -> dict:
        """Validate if retrieved context is sufficient"""
        # Check relevance, completeness, etc.
        is_sufficient = check_context_quality(query, context)
        return {"sufficient": is_sufficient}
    
    agent = create_agent(
        model=llm,
        tools=[retrieve_with_enhancement, validate_retrieval],
        system_prompt="Retrieve and validate context before generating questions."
    )
    return agent
```

### **2. Multi-Source RAG**

Combine multiple vector stores or sources:

```python
@tool
def retrieve_from_pdf(query: str) -> str:
    """Retrieve from PDF knowledge base"""
    return pdf_vector_store.similarity_search(query, k=3)

@tool
def retrieve_from_web(query: str) -> str:
    """Retrieve from web sources"""
    return web_vector_store.similarity_search(query, k=3)

agent = create_agent(
    model=llm,
    tools=[retrieve_from_pdf, retrieve_from_web],
    system_prompt="Retrieve from multiple sources for comprehensive question generation."
)
```

### **3. Streaming RAG**

Stream responses for better UX:

```python
rag_agent = setup_rag_agent(vector_store)

for event in rag_agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values"
):
    print(event["messages"][-1].content)
```

---

## 📚 Updated Dependencies

### **Required Packages (setup.py):**

```python
install_requires=[
    # Core LangChain v1.0
    "langchain>=1.0.0",
    "langchain-core>=1.0.0",
    "langchain-openai>=0.2.0",
    "langchain-community>=0.3.0",
    "langchain-text-splitters>=0.3.0",
    
    # Provider-specific
    "langchain-google-genai",
    
    # Vector stores and embeddings
    "chromadb",
    "openai",
    
    # Other dependencies
    "pydantic",
    "python-dotenv",
    # ... rest of dependencies
]
```

### **Remove:**
```python
"langchain-classic",  # ❌ Deprecated - remove this
```

---

## ✅ Migration Checklist

### **Code Changes**
- [ ] Remove `from langchain_classic.chains.retrieval_qa.base import RetrievalQA`
- [ ] Add modern imports (`create_agent`, `tool`, etc.)
- [ ] Implement `_create_retrieval_tool()` method
- [ ] Implement `_setup_rag_agent()` method
- [ ] Update `generate_questions_with_rag()` to use agent
- [ ] Remove `_setup_retrieval_qa()` method

### **Dependencies**
- [ ] Remove `langchain-classic` from `setup.py`
- [ ] Update `langchain` to `>=1.0.0`
- [ ] Update other langchain packages to compatible versions

### **Testing**
- [ ] Test with PDF sources
- [ ] Test with URL sources
- [ ] Test with text sources
- [ ] Test all question types (MCQ, Short Answer, True/False, Fill in Blank)
- [ ] Test with learning objectives
- [ ] Test with difficulty levels
- [ ] Test output formats (PDF, CSV)
- [ ] Performance testing (compare speed)
- [ ] Quality testing (compare question quality)

### **Documentation**
- [ ] Update README with new RAG approach
- [ ] Add examples using the new pattern
- [ ] Document migration for users
- [ ] Update API documentation

---

## 🚀 Quick Start: Minimal Migration

If you want the **absolute minimum changes** to get off `langchain-classic`:

```python
# qna_engine.py - MINIMAL MIGRATION

# 1. Change import
from langchain.agents import create_agent
from langchain.tools import tool

# 2. Replace _setup_retrieval_qa with this:
def _setup_rag_agent(self, vector_store):
    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve relevant context"""
        docs = vector_store.similarity_search(query, k=4)
        return "\n\n".join(doc.page_content for doc in docs)
    
    return create_agent(
        model=self.llm,
        tools=[retrieve_context],
        system_prompt="Use the retrieve_context tool to get information before generating questions."
    )

# 3. In generate_questions_with_rag, replace:
# qa_chain = self._setup_retrieval_qa(vector_store)
# with:
rag_agent = self._setup_rag_agent(vector_store)

# 4. Update the invocation:
response = rag_agent.invoke({
    "messages": [{"role": "user", "content": your_query}]
})
result = response["messages"][-1].content
```

**That's it!** You're now using modern LangChain v1.0 with no deprecated code.

---

## 🔮 Future Enhancements

Once migrated, you can easily add:

1. **Web search tool** for real-time information
2. **Calculator tool** for math questions
3. **Image generation** for visual questions
4. **Multi-modal RAG** with images
5. **Streaming responses** for better UX
6. **Query enhancement** for better retrieval
7. **Answer validation** for quality control
8. **Multiple vector stores** for diverse content

---

## 📖 Resources

- [LangChain v1.0 RAG Tutorial](https://docs.langchain.com/oss/python/langchain/rag)
- [LangChain v1.0 Agents Guide](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain v1.0 Tools Documentation](https://docs.langchain.com/oss/python/langchain/tools)
- [LangChain v1.0 Retrieval Overview](https://docs.langchain.com/oss/python/langchain/retrieval)
- [Agentic RAG Tutorial](https://docs.langchain.com/oss/python/langgraph/agentic-rag)

---

## 🎓 Summary

### **What to Do:**
1. ✅ Replace `RetrievalQA` with RAG Agent pattern
2. ✅ Use `@tool` decorator for retrieval
3. ✅ Use `create_agent()` for agent creation
4. ✅ Remove `langchain-classic` dependency

### **What to Avoid:**
- ❌ Don't use `langchain-classic` package
- ❌ Don't use `RetrievalQA` chain
- ❌ Don't use legacy chain patterns

### **Benefits:**
- ✅ Future-proof code
- ✅ Better flexibility
- ✅ More powerful features
- ✅ Easier to extend
- ✅ Aligned with LangChain direction

---

**Document Version**: 1.0  
**Date**: November 18, 2024  
**Purpose**: Complete migration guide from langchain-classic to LangChain v1.0 modern patterns
