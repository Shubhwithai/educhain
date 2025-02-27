# in educhain/models/qna_models.py
from educhain.models.base_models import BaseQuestion, QuestionList
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class MultipleChoiceQuestion(BaseQuestion):
    options: List[str]

    def show(self):
        print(f"Question: {self.question}")
        options_str = "\n".join(f"  {chr(65 + i)}. {option}" for i, option in enumerate(self.options))
        print(f"Options:\n{options_str}")
        print(f"\nCorrect Answer: {self.answer}")
        if self.explanation:
            print(f"Explanation: {self.explanation}")
        print()

# Add these new models:
class GraphInstruction(BaseModel):
    type: str = Field(..., description="Type of visualization (bar, pie, line, scatter, table)")
    x_labels: Optional[List[str]] = Field(None, description="Labels for x-axis (for bar, line)")
    x_values: Optional[List[Any]] = Field(None, description="Values for x-axis (for scatter)")
    y_values: Optional[List[Any]] = Field(None, description="Values for y-axis (for bar, line, scatter, multiple lines in line)")
    labels: Optional[List[str]] = Field(None, description="Labels for pie chart segments or line graph legend")
    sizes: Optional[List[float]] = Field(None, description="Sizes for pie chart segments")
    y_label: Optional[str] = Field(None, description="Label for y-axis")
    title: Optional[str] = Field(None, description="Title of the visualization")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Data for table visualization")


class VisualMCQ(MultipleChoiceQuestion):
    graph_instruction: Optional[GraphInstruction] = Field(None, description="Instructions for generating a graph")

    def show(self):
        super().show()
        if self.graph_instruction:
            print(f"Graph Instruction: {self.graph_instruction}")
        print()

class VisualMCQList(QuestionList):
    questions: List[VisualMCQ]


class ShortAnswerQuestion(BaseQuestion):
    keywords: List[str] = Field(default_factory=list)

    def show(self):
        super().show()
        if self.keywords:
            print(f"Keywords: {', '.join(self.keywords)}")
        print()

class TrueFalseQuestion(BaseQuestion):
    answer: bool

    def show(self):
        super().show()
        print(f"True/False: {self.answer}")
        print()

class FillInBlankQuestion(BaseQuestion):
    blank_word: Optional[str] = None

    def show(self):
        super().show()
        print(f"Word to fill: {self.blank_word or self.answer}")
        print()

class MCQList(QuestionList):
    questions: List[MultipleChoiceQuestion]

class ShortAnswerQuestionList(QuestionList):
    questions: List[ShortAnswerQuestion]

class TrueFalseQuestionList(QuestionList):
    questions: List[TrueFalseQuestion]

class FillInBlankQuestionList(QuestionList):
    questions: List[FillInBlankQuestion]

class Option(BaseModel):
    text: str = Field(description="The text of the option.")
    correct: str = Field(description="Whether the option is correct or not. Either 'true' or 'false'")

class MCQMath(BaseModel):
    question: str = Field(description="The quiz question, strictly avoid Latex formatting")
    requires_math: bool = Field(default=False, description="Whether the question requires the LLM Math Chain for accurate answers.")
    options: List[Option] = Field(description="The possible answers to the question. The list should contain 4 options.")
    explanation: str =  Field(default=None, description="Explanation of the question")

    def show(self):
        print(f"Question: {self.question}")
        for i, option in enumerate(self.options):
            print(f"  {chr(65 + i)}. {option.text} {'(Correct)' if option.correct == 'true' else ''}")
        if self.explanation:
            print(f"Explanation: {self.explanation}")
        print()

class MCQListMath(BaseModel):
    questions: List[MCQMath]

    def show(self):
        for i, question in enumerate(self.questions, 1):
            print(f"Question {i}:")
            question.show()

class SolvedDoubt(BaseModel):
    """Model for representing a solved doubt with explanation and steps"""
    explanation: str = Field(
        description="Detailed explanation of the problem and its solution"
    )
    steps: List[str] = Field(
        default_factory=list,
        description="Step-by-step solution process"
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description="Additional tips, warnings, or relevant information"
    )

    def show(self):
        """Display the solved doubt in a formatted way"""
        print("\n=== Problem Explanation ===")
        print(self.explanation)

        if self.steps:
            print("\n=== Solution Steps ===")
            for i, step in enumerate(self.steps, 1):
                print(f"{i}. {step}")

        if self.additional_notes:
            print("\n=== Additional Notes ===")
            print(self.additional_notes)

class SpeechInstructions(BaseModel):
    topic: str
    num_questions: Optional[int] = 5
    custom_instructions: Optional[str] = None
    detected_language: Optional[str] = "english"

class RAGQuestion(BaseQuestion):
    """Model for questions generated using RAG (Retrieval Augmented Generation)"""
    source_context: Optional[str] = Field(
        default=None,
        description="The relevant context from which this question was generated"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="Confidence score for the generated question (0-1)"
    )
    difficulty_level: Optional[str] = Field(
        default=None,
        description="Difficulty level of the question"
    )
    learning_objective: Optional[str] = Field(
        default=None,
        description="Learning objective this question addresses"
    )
    chunk_index: Optional[int] = Field(
        default=None,
        description="Index of the content chunk used to generate this question"
    )
    options: List[str] = Field(
        default_factory=list,
        description="List of options for multiple choice questions"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata about the question generation"
    )

    def show(self):
        print(f"Question: {self.question}")
        if self.options:
            options_str = "\n".join(f"  {chr(65 + i)}. {option}" for i, option in enumerate(self.options))
            print(f"Options:\n{options_str}")
        print(f"\nCorrect Answer: {self.answer}")
        if self.explanation:
            print(f"Explanation: {self.explanation}")
        if self.difficulty_level:
            print(f"Difficulty: {self.difficulty_level}")
        if self.learning_objective:
            print(f"Learning Objective: {self.learning_objective}")
        if self.source_context:
            print(f"\nSource Context: {self.source_context}")
        if self.confidence_score is not None:
            print(f"Confidence Score: {self.confidence_score:.2f}")
        print()

class RAGQuestionList(QuestionList):
    """List of questions generated using RAG"""
    questions: List[RAGQuestion]
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the overall question generation process"
    )
    retrieval_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Statistics about the retrieval process"
    )
    generation_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration used for generation"
    )

    def show(self):
        print("=== RAG Question Generation Report ===")
        if self.metadata:
            print("\n=== Generation Metadata ===")
            for key, value in self.metadata.items():
                print(f"{key}: {value}")
        
        if self.retrieval_stats:
            print("\n=== Retrieval Statistics ===")
            for key, value in self.retrieval_stats.items():
                print(f"{key}: {value}")
        
        if self.generation_config:
            print("\n=== Generation Configuration ===")
            for key, value in self.generation_config.items():
                print(f"{key}: {value}")
        print("\n=== Generated Questions ===")
        
        for i, question in enumerate(self.questions, 1):
            print(f"\nQuestion {i}:")
            question.show()

class DataSourceQuestion(BaseQuestion):
    """Base model for questions generated from specific data sources"""
    source_type: str = Field(
        description="Type of source the question was generated from (pdf, url, text)"
    )
    source_location: Optional[str] = Field(
        default=None,
        description="Location or identifier of the source"
    )
    content_segment: Optional[str] = Field(
        default=None,
        description="Relevant segment of the source content"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata about the question and its source"
    )

class DataSourceMCQ(DataSourceQuestion):
    """Multiple Choice Question from data source"""
    options: List[str]

    def show(self):
        print(f"Question: {self.question}")
        options_str = "\n".join(f"  {chr(65 + i)}. {option}" for i, option in enumerate(self.options))
        print(f"Options:\n{options_str}")
        print(f"\nCorrect Answer: {self.answer}")
        if self.explanation:
            print(f"Explanation: {self.explanation}")
        print(f"\nSource Type: {self.source_type}")
        if self.source_location:
            print(f"Source: {self.source_location}")
        if self.content_segment:
            print(f"\nRelevant Content:\n{self.content_segment}")
        print()

class DataSourceQuestionList(QuestionList):
    """List of questions generated from data sources"""
    questions: List[str]
    source_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the source and generation process"
    )
    processing_stats: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Statistics about the content processing"
    )

    def show(self):
        if self.source_metadata:
            print("=== Source Information ===")
            for key, value in self.source_metadata.items():
                print(f"{key}: {value}")
            print()
            
        if self.processing_stats:
            print("=== Processing Statistics ===")
            for key, value in self.processing_stats.items():
                print(f"{key}: {value}")
            print()
        
        for i, question in enumerate(self.questions, 1):
            print(f"Question {i}:")
            question.show()

class DataSourceMultipleChoiceQuestion(MultipleChoiceQuestion):
    """Multiple Choice Question with data source information"""
    source_type: str = Field(
        description="Type of source the question was generated from (pdf, url, text)"
    )
    source_location: Optional[str] = Field(
        default=None,
        description="Location or identifier of the source"
    )
    content_segment: Optional[str] = Field(
        default=None,
        description="Relevant segment of the source content"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata about the question and its source"
    )

    def show(self):
        # First show the basic MCQ information
        super().show()
        # Then show the source information
        print(f"Source Type: {self.source_type}")
        if self.source_location:
            print(f"Source: {self.source_location}")
        if self.content_segment:
            print(f"Relevant Content:\n{self.content_segment}")
        if self.metadata:
            print("Additional Metadata:")
            for key, value in self.metadata.items():
                print(f"  {key}: {value}")
        print()

class DataSourceMCQList(MCQList):
    """List of Multiple Choice Questions from data sources"""
    questions: List[DataSourceMultipleChoiceQuestion]
    source_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the source and generation process"
    )
    processing_stats: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Statistics about the content processing"
    )

    def show(self):
        if self.source_metadata:
            print("=== Source Information ===")
            for key, value in self.source_metadata.items():
                print(f"{key}: {value}")
            print()
            
        if self.processing_stats:
            print("=== Processing Statistics ===")
            for key, value in self.processing_stats.items():
                print(f"{key}: {value}")
            print()
        
        for i, question in enumerate(self.questions, 1):
            print(f"Question {i}:")
            question.show()

class BaseSourceQuestion(BaseQuestion):
    """Base model for questions generated from any source (RAG or direct data)"""
    source_type: str = Field(
        description="Type of source (pdf, url, text)"
    )
    source_context: Optional[str] = Field(
        default=None,
        description="Relevant context or segment from source"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the question"
    )

    def show(self):
        print(f"Question: {self.question}")
        print(f"Answer: {self.answer}")
        if self.explanation:
            print(f"Explanation: {self.explanation}")
        print(f"Source Type: {self.source_type}")
        if self.source_context:
            print(f"Context: {self.source_context}")
        print()

class SourceMCQ(BaseSourceQuestion):
    """Multiple Choice Question with source information"""
    options: List[str]
    
    def show(self):
        print(f"Question: {self.question}")
        for i, option in enumerate(self.options):
            print(f"  {chr(65 + i)}. {option}")
        print(f"\nCorrect Answer: {self.answer}")
        if self.explanation:
            print(f"Explanation: {self.explanation}")
        print(f"Source Type: {self.source_type}")
        if self.source_context:
            print(f"Context: {self.source_context}")
        print()

class SourceQuestionList(QuestionList):
    """Base list model for questions generated from any source"""
    questions: List[BaseSourceQuestion]
    generation_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Statistics about the generation process"
    )

    def show(self):
        if self.generation_stats:
            print("=== Generation Statistics ===")
            for key, value in self.generation_stats.items():
                print(f"{key}: {value}")
            print()
        
        for i, question in enumerate(self.questions, 1):
            print(f"\nQuestion {i}:")
            question.show()

class SourceMCQList(SourceQuestionList):
    """List of Multiple Choice Questions from any source"""
    questions: List[SourceMCQ]