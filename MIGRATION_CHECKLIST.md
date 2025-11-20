# ✅ Educhain LangChain v1.0 Migration Checklist

## 🎯 Goal
Remove all deprecated `langchain-classic` code and migrate to modern LangChain v1.0 patterns.

---

## 📋 Quick Summary

| Item | Status | Priority |
|------|--------|----------|
| Remove Python 3.9 support | ⚠️ Required | 🔴 High |
| Replace RetrievalQA with RAG Agent | ⚠️ Required | 🔴 High |
| Remove langchain-classic dependency | ⚠️ Required | 🔴 High |
| Update package versions | ⚠️ Required | 🔴 High |
| Adopt content_blocks | 🔄 Optional | 🟡 Medium |
| Use init_chat_model | 🔄 Optional | 🟢 Low |

---

## 🔴 CRITICAL: Required Changes

### 1. **setup.py Changes**

#### ❌ Remove:
```python
# Line 53
"Programming Language :: Python :: 3.9",

# Line 13
"langchain-classic",  # Deprecated package
```

#### ✅ Update:
```python
# Update versions
install_requires=[
    "langchain>=1.0.0",
    "langchain-core>=1.0.0",
    "langchain-text-splitters>=0.3.0",
    "langchain-community>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-google-genai",
    # ... rest remains same
]

# Update classifiers
classifiers=[
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",  # Keep
    "Programming Language :: Python :: 3.11",  # Keep
    "Programming Language :: Python :: 3.12",  # Keep
],
python_requires='>=3.10',  # Already correct
```

### 2. **educhain/engines/qna_engine.py Changes**

#### ❌ Remove Import (Line 17):
```python
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
```

#### ✅ Add New Imports (Top of file):
```python
from langchain.agents import create_agent
from langchain.tools import tool
```

#### ❌ Remove Method:
```python
def _setup_retrieval_qa(self, vector_store: Chroma) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=self.llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )
```

#### ✅ Add New Methods:
```python
def _create_retrieval_tool(self, vector_store):
    """Create a retrieval tool for the agent"""
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str) -> str:
        """Retrieve relevant context from the knowledge base."""
        retrieved_docs = vector_store.similarity_search(query, k=4)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    return retrieve_context

def _setup_rag_agent(self, vector_store):
    """Setup RAG agent using modern LangChain v1.0 pattern"""
    retrieve_tool = self._create_retrieval_tool(vector_store)
    
    system_prompt = """
    You are an expert educational content generator.
    Use the retrieve_context tool to get relevant information before generating questions.
    Generate clear, accurate, and pedagogically sound questions.
    """
    
    agent = create_agent(
        model=self.llm,
        tools=[retrieve_tool],
        system_prompt=system_prompt
    )
    return agent
```

#### ✅ Update generate_questions_with_rag Method:

**Find line ~474:**
```python
qa_chain = self._setup_retrieval_qa(vector_store)
```

**Replace with:**
```python
rag_agent = self._setup_rag_agent(vector_store)
```

**Update invocation pattern:**
```python
# Old pattern (around line 500-520):
# Build prompt and invoke qa_chain

# New pattern:
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

query += f"\n\nOutput Format:\n{format_instructions}"

# Invoke the agent
response = rag_agent.invoke({
    "messages": [{"role": "user", "content": query}]
})

# Extract the result
result_content = response["messages"][-1].content
```

---

## 🔄 Optional Enhancements (Future-Proofing)

### 1. **Use content_blocks (Recommended)**

In both `qna_engine.py` and `content_engine.py`, when accessing LLM responses:

**Current:**
```python
results = llm.invoke(messages)
content = results.content  # String
```

**Enhanced (for multimodal support):**
```python
results = llm.invoke(messages)

# Extract text from content blocks
content = ""
for block in results.content_blocks:
    if block["type"] == "text":
        content += block.get("text", "")
```

### 2. **Use init_chat_model (Optional)**

**Current in _initialize_llm():**
```python
return ChatOpenAI(
    model=llm_config.model_name,
    api_key=llm_config.api_key,
    # ...
)
```

**Modern (provider-agnostic):**
```python
from langchain.chat_models import init_chat_model

return init_chat_model(
    model=llm_config.model_name,
    model_provider="openai",
    api_key=llm_config.api_key,
    # ...
)
```

---

## 🧪 Testing Checklist

After making changes, test:

### Core Functionality
- [ ] `generate_questions()` - all question types
- [ ] `generate_questions_from_data()` - PDF source
- [ ] `generate_questions_from_data()` - URL source
- [ ] `generate_questions_from_data()` - text source
- [ ] `generate_questions_with_rag()` - PDF source ⭐ **CRITICAL**
- [ ] `generate_questions_with_rag()` - URL source ⭐ **CRITICAL**
- [ ] `generate_questions_with_rag()` - text source ⭐ **CRITICAL**

### Question Types
- [ ] Multiple Choice Questions (MCQ)
- [ ] Short Answer Questions
- [ ] True/False Questions
- [ ] Fill in the Blank Questions
- [ ] Math MCQ
- [ ] Visual MCQ

### Parameters
- [ ] Learning objectives
- [ ] Difficulty levels
- [ ] Custom instructions
- [ ] Custom prompt templates
- [ ] Response models

### Output Formats
- [ ] JSON output (default)
- [ ] PDF output
- [ ] CSV output

### Content Generation
- [ ] Lesson plans
- [ ] Study guides
- [ ] Flashcards
- [ ] Career connections
- [ ] Podcast scripts
- [ ] Pedagogy-based content

### Edge Cases
- [ ] Large documents
- [ ] Empty content
- [ ] Invalid sources
- [ ] API errors
- [ ] Parsing errors

---

## 📦 Installation & Setup

### 1. Update Dependencies
```bash
cd /Users/shubham/Documents/BFWAI\ Main\ Projects\ /educhain
pip install --upgrade langchain langchain-core langchain-openai langchain-community
pip uninstall langchain-classic  # Remove deprecated package
```

### 2. Test Import
```python
python -c "from langchain.agents import create_agent; print('✅ LangChain v1.0 imports working')"
```

### 3. Run Tests
```bash
# If you have tests
pytest tests/

# Or manual testing
python -c "from educhain import QnAEngine; print('✅ Educhain imports working')"
```

---

## 📝 File-by-File Changes

### **File 1: setup.py**
- [ ] Line 13: Remove `"langchain-classic"`
- [ ] Line 8-14: Update langchain package versions
- [ ] Line 53: Remove Python 3.9 classifier

### **File 2: educhain/engines/qna_engine.py**
- [ ] Line 17: Remove `RetrievalQA` import
- [ ] Add imports: `create_agent`, `tool`
- [ ] Add method: `_create_retrieval_tool()`
- [ ] Add method: `_setup_rag_agent()`
- [ ] Remove method: `_setup_retrieval_qa()`
- [ ] Update: `generate_questions_with_rag()` implementation

### **File 3: educhain/engines/content_engine.py**
- [ ] No required changes (already compatible)
- [ ] Optional: Consider using `content_blocks`

---

## 🎯 Success Criteria

✅ **Migration Complete When:**
1. No imports from `langchain-classic`
2. All tests passing
3. RAG question generation working
4. No deprecation warnings
5. Package published with v1.0 support

---

## 🚨 Common Issues & Solutions

### Issue 1: Import Error
```
ImportError: cannot import name 'RetrievalQA' from 'langchain_classic'
```
**Solution:** You forgot to add the new imports or didn't replace the method.

### Issue 2: Agent Not Working
```
AttributeError: 'dict' object has no attribute 'content'
```
**Solution:** Agent returns different structure. Access: `response["messages"][-1].content`

### Issue 3: Tool Not Called
```
Agent generates questions without retrieving context
```
**Solution:** Improve system prompt to explicitly instruct tool usage.

---

## 📚 Documentation to Update

- [ ] README.md - Update examples with new RAG pattern
- [ ] API documentation
- [ ] Tutorial notebooks
- [ ] Example scripts in `/cookbook`
- [ ] Migration guide for users

---

## 🔗 Reference Documents

1. **LANGCHAIN_V1_MIGRATION_ANALYSIS.md** - Complete analysis
2. **LANGCHAIN_V1_MODERN_REPLACEMENTS.md** - Detailed code examples
3. **MIGRATION_CHECKLIST.md** - This file

---

## ⏱️ Estimated Time

- **Required Changes**: 2-3 hours
- **Testing**: 2-3 hours
- **Documentation**: 1-2 hours
- **Total**: 5-8 hours

---

## 🎉 Next Steps After Migration

1. ✅ Commit changes with clear message
2. ✅ Update version number (e.g., 0.4.0 for major changes)
3. ✅ Test in production-like environment
4. ✅ Publish updated package to PyPI
5. ✅ Announce migration in docs/blog
6. ✅ Monitor for issues

---

**Last Updated**: November 18, 2024  
**Migration Target**: LangChain v1.0+  
**Priority**: HIGH - Complete ASAP to avoid deprecation issues
