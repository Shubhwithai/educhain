# Pydantic v2 Migration Guide for Educhain

## 🎯 Current Status

**Pydantic Version Installed:** v2.11.10 ✅  
**Code Compatibility:** Using deprecated v1 patterns ⚠️  
**Migration Required:** YES 🔴

---

## 🚨 Issues Found

Your codebase is using **Pydantic v2.11.10** but still has **deprecated Pydantic v1** patterns that will cause warnings and may break in future versions.

### **Deprecated Patterns Found:**

| Pattern | Count | Status | Replacement |
|---------|-------|--------|-------------|
| `.dict()` | 15+ occurrences | ⚠️ Deprecated | `.model_dump()` |
| `__fields__` | 5 occurrences | ⚠️ Deprecated | `model_fields` |
| No version constraint | 1 | ⚠️ Risk | Add `pydantic>=2.0` |

---

## 📋 Pydantic v1 vs v2 Changes

### **1. Model Serialization**

#### ❌ **Pydantic v1 (Deprecated)**
```python
# To dictionary
data_dict = model.dict()

# To JSON string
json_str = model.json()

# From dictionary
model = Model.parse_obj(data)

# From JSON
model = Model.parse_raw(json_str)
```

#### ✅ **Pydantic v2 (Current)**
```python
# To dictionary
data_dict = model.model_dump()

# To JSON string
json_str = model.model_dump_json()

# From dictionary
model = Model.model_validate(data)

# From JSON
model = Model.model_validate_json(json_str)
```

### **2. Model Fields Access**

#### ❌ **Pydantic v1 (Deprecated)**
```python
# Access fields
fields = Model.__fields__

# Check if field exists
if 'field_name' in Model.__fields__:
    ...

# Get field info
field_info = Model.__fields__['field_name']
```

#### ✅ **Pydantic v2 (Current)**
```python
# Access fields
fields = Model.model_fields

# Check if field exists
if 'field_name' in Model.model_fields:
    ...

# Get field info
field_info = Model.model_fields['field_name']
```

### **3. Model Configuration**

#### ❌ **Pydantic v1 (Deprecated)**
```python
class Model(BaseModel):
    field: str
    
    class Config:
        arbitrary_types_allowed = True
        orm_mode = True
```

#### ✅ **Pydantic v2 (Current)**
```python
from pydantic import ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True  # replaces orm_mode
    )
    
    field: str
```

### **4. Validators**

#### ❌ **Pydantic v1 (Deprecated)**
```python
from pydantic import validator

class Model(BaseModel):
    email: str
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v
```

#### ✅ **Pydantic v2 (Current)**
```python
from pydantic import field_validator

class Model(BaseModel):
    email: str
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v
```

---

## 🔍 Files Requiring Updates

### **1. setup.py**
**Issue:** No Pydantic version constraint  
**Line:** 20  
**Fix:** Add version requirement

### **2. educhain/engines/qna_engine.py**
**Issues:**
- `.dict()` usage: Lines 369, 979, 1037, 1178, 1566, 1677, 1756
- `__fields__` usage: Lines 995, 1066-1068, 1196

### **3. educhain/utils/output_formatter.py**
**Issues:**
- `.dict()` usage: Lines 20, 23, 26

### **4. Models (No issues found - already compatible)** ✅
- `educhain/models/base_models.py` - Clean ✅
- `educhain/models/qna_models.py` - Clean ✅
- `educhain/models/content_models.py` - Clean ✅
- `educhain/models/pedagogy_models.py` - Clean ✅

---

## 🔧 Detailed Migration Steps

### **Step 1: Update setup.py**

**Current:**
```python
"pydantic",  # Line 20
```

**Updated:**
```python
"pydantic>=2.0,<3.0",  # Require Pydantic v2
```

---

### **Step 2: Update output_formatter.py**

**File:** `educhain/utils/output_formatter.py`

**Lines 20, 23, 26 - Replace `.dict()` with `.model_dump()`:**

**Current:**
```python
def _convert_to_dict_list(data: Any) -> List[Dict]:
    """Convert Pydantic model data to a list of dictionaries"""
    if hasattr(data, 'questions'):
        # If it's a question list model
        return [q.dict() for q in data.questions]  # Line 20
    elif isinstance(data, list):
        # If it's already a list
        return [item.dict() if hasattr(item, 'dict') else item for item in data]  # Line 23
    else:
        # Single item
        return [data.dict() if hasattr(data, 'dict') else data]  # Line 26
```

**Updated:**
```python
def _convert_to_dict_list(data: Any) -> List[Dict]:
    """Convert Pydantic model data to a list of dictionaries"""
    if hasattr(data, 'questions'):
        # If it's a question list model
        return [q.model_dump() for q in data.questions]
    elif isinstance(data, list):
        # If it's already a list
        return [item.model_dump() if hasattr(item, 'model_dump') else item for item in data]
    else:
        # Single item
        return [data.model_dump() if hasattr(data, 'model_dump') else data]
```

---

### **Step 3: Update qna_engine.py**

**File:** `educhain/engines/qna_engine.py`

#### **A. Replace `.dict()` method calls**

**Line 369:**
```python
# Current
self._generate_and_save_visual(instruction.dict(), question_text, options, correct_answer)

# Updated
self._generate_and_save_visual(instruction.model_dump(), question_text, options, correct_answer)
```

**Line 979:**
```python
# Current
q_dict = question.dict() if hasattr(question, 'dict') else question

# Updated
q_dict = question.model_dump() if hasattr(question, 'model_dump') else question
```

**Line 1037:**
```python
# Current
if hasattr(value, 'dict'):
    value = value.dict()

# Updated
if hasattr(value, 'model_dump'):
    value = value.model_dump()
```

**Line 1178:**
```python
# Current
question_dict = question.dict() if hasattr(question, 'dict') else question

# Updated
question_dict = question.model_dump() if hasattr(question, 'model_dump') else question
```

**Line 1566:**
```python
# Current
q_dict = question.dict() if hasattr(question, 'dict') else question

# Updated
q_dict = question.model_dump() if hasattr(question, 'model_dump') else question
```

**Line 1677:**
```python
# Current
elif hasattr(option, 'dict'):  # Pydantic model
    option_dict = option.dict()

# Updated
elif hasattr(option, 'model_dump'):  # Pydantic model
    option_dict = option.model_dump()
```

**Line 1756:**
```python
# Current
json.dump([q.dict() if hasattr(q, 'dict') else q for q in all_questions], f, indent=4)

# Updated
json.dump([q.model_dump() if hasattr(q, 'model_dump') else q for q in all_questions], f, indent=4)
```

#### **B. Replace `__fields__` with `model_fields`**

**Line 995:**
```python
# Current
if 'metadata' not in q_dict and hasattr(question_model, '__fields__') and 'metadata' in question_model.__fields__:

# Updated
if 'metadata' not in q_dict and hasattr(question_model, 'model_fields') and 'metadata' in question_model.model_fields:
```

**Lines 1066-1068:**
```python
# Current
required_fields = {field for field, _ in question_model.__fields__.items() 
                 if not question_model.__fields__[field].default_factory
                 and question_model.__fields__[field].default is None}

# Updated
required_fields = {field for field, field_info in question_model.model_fields.items() 
                 if field_info.is_required()}
```

**Line 1196:**
```python
# Current
if 'metadata' not in question_dict and hasattr(question_model, '__fields__') and 'metadata' in question_model.__fields__:

# Updated
if 'metadata' not in question_dict and hasattr(question_model, 'model_fields') and 'metadata' in question_model.model_fields:
```

---

## 🔄 Complete Migration Script

Here's a helper function you can add to ensure backward compatibility during transition:

```python
# Add to utils/helpers.py or similar

def safe_model_dump(model):
    """
    Safely dump a Pydantic model to dict, supporting both v1 and v2
    """
    if hasattr(model, 'model_dump'):
        # Pydantic v2
        return model.model_dump()
    elif hasattr(model, 'dict'):
        # Pydantic v1 (deprecated but still works)
        return model.dict()
    else:
        # Not a Pydantic model
        return model

def safe_model_fields(model_class):
    """
    Safely get model fields, supporting both v1 and v2
    """
    if hasattr(model_class, 'model_fields'):
        # Pydantic v2
        return model_class.model_fields
    elif hasattr(model_class, '__fields__'):
        # Pydantic v1 (deprecated)
        return model_class.__fields__
    else:
        return {}
```

---

## ✅ Migration Checklist

### **Required Changes:**
- [ ] Update `setup.py` - Add Pydantic version constraint
- [ ] Update `output_formatter.py` - Replace `.dict()` with `.model_dump()`
- [ ] Update `qna_engine.py` - Replace all `.dict()` calls (7 occurrences)
- [ ] Update `qna_engine.py` - Replace all `__fields__` usage (3 occurrences)

### **Testing:**
- [ ] Test question generation (all types)
- [ ] Test CSV output format
- [ ] Test PDF output format
- [ ] Test JSON output format
- [ ] Test bulk question generation
- [ ] Test RAG-based generation
- [ ] Verify no deprecation warnings

### **Documentation:**
- [ ] Update README if needed
- [ ] Update CHANGELOG
- [ ] Note Pydantic v2 requirement

---

## 🧪 Testing After Migration

### **1. Test Basic Functionality**
```python
from educhain import QnAEngine

qna = QnAEngine()

# Test question generation
questions = qna.generate_questions(
    topic="Python programming",
    num=3,
    question_type="Multiple Choice"
)

# Test serialization (Pydantic v2)
for q in questions.questions:
    q_dict = q.model_dump()  # Should work
    print(q_dict)
```

### **2. Test Output Formats**
```python
# Test CSV output
qna.generate_questions(
    topic="Data Science",
    num=5,
    output_format="csv"
)

# Test PDF output
qna.generate_questions(
    topic="Machine Learning",
    num=5,
    output_format="pdf"
)
```

### **3. Check for Deprecation Warnings**
```python
import warnings
warnings.filterwarnings('error', category=DeprecationWarning)

# Run your code - should not raise any warnings
```

---

## 📊 Migration Impact

### **Breaking Changes:**
- ⚠️ Requires Pydantic v2.0+ (if you update setup.py constraint)
- ⚠️ Code using `.dict()` will show deprecation warnings (still works but will be removed)

### **Benefits:**
- ✅ **Performance:** Pydantic v2 is 5-50x faster than v1
- ✅ **Type Safety:** Better type checking and validation
- ✅ **Features:** Access to new v2 features
- ✅ **Future-proof:** No deprecation warnings
- ✅ **Maintenance:** Aligned with current best practices

### **Compatibility:**
- ✅ **Models:** Already compatible (no Config class, no @validator)
- ✅ **BaseModel:** Works with both v1 and v2
- ✅ **Field:** Works with both v1 and v2
- ⚠️ **Methods:** Need to update `.dict()` → `.model_dump()`

---

## 🚀 Quick Migration Command

Create this helper script to find all occurrences:

```bash
#!/bin/bash
# find_pydantic_v1.sh

echo "Finding Pydantic v1 patterns..."
echo ""

echo "=== .dict() calls ==="
grep -rn "\.dict()" educhain/ --include="*.py" | grep -v "# Updated" | grep -v "model_dump"

echo ""
echo "=== __fields__ usage ==="
grep -rn "__fields__" educhain/ --include="*.py" | grep -v "model_fields"

echo ""
echo "=== parse_obj usage ==="
grep -rn "parse_obj" educhain/ --include="*.py"

echo ""
echo "=== Done ==="
```

---

## 📚 Resources

- [Pydantic v2 Migration Guide](https://docs.pydantic.dev/latest/migration/)
- [Pydantic v2 Release Notes](https://docs.pydantic.dev/latest/changelog/)
- [Pydantic v2 Performance Improvements](https://docs.pydantic.dev/latest/concepts/performance/)

---

## 🎯 Estimated Migration Time

- **Code Changes:** 1-2 hours
- **Testing:** 1-2 hours
- **Total:** 2-4 hours

---

## ⚠️ Important Notes

1. **Current State:** Code works but shows deprecation warnings
2. **Future Risk:** `.dict()` and `__fields__` will be removed in Pydantic v3
3. **Recommended:** Migrate now to avoid breaking changes later
4. **Priority:** MEDIUM - Works now but should be updated soon

---

**Document Version:** 1.0  
**Date:** November 21, 2024  
**Status:** Ready for Implementation
