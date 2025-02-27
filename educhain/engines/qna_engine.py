# educhain/engines/qna_engine.py

from typing import Optional, Type, Any, List, Literal, Union, Tuple
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA, LLMMathChain
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks.manager import get_openai_callback
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import re
from langchain_core.messages import SystemMessage
from langchain.schema import HumanMessage
from educhain.core.config import LLMConfig
from educhain.models.qna_models import (
    MCQList, ShortAnswerQuestionList, TrueFalseQuestionList,
    FillInBlankQuestionList, MCQListMath, Option ,SolvedDoubt, SpeechInstructions,
    VisualMCQList, VisualMCQ, DataSourceMCQList
)
from educhain.utils.loaders import PdfFileLoader, UrlLoader
from educhain.utils.output_formatter import OutputFormatter
import base64
import os
from PIL import Image
import io
import logging
import matplotlib.pyplot as plt
import pandas as pd
import dataframe_image as dfi
from IPython.display import display, HTML
import random
import time

QuestionType = Literal["Multiple Choice", "Short Answer", "True/False", "Fill in the Blank"]
OutputFormatType = Literal["pdf", "csv"]

VISUAL_QUESTION_PROMPT_TEMPLATE = """Generate exactly {num} quantitative questions based on the topic: {topic}.
        Each question should require a visual representation of the data (bar graph, pie chart, line graph, or scatter plot or table) along with a detailed instruction on how to create that visual and options for the question. The question should be solvable based on the data in the visual.

        The visual type should be chosen based on the topic.
        Here is the general guidance for the visualization type:
        - Use pie chart when visualizing proportions or parts of a whole.
        - Use bar or column chart for comparing discrete categories or for displaying the frequency distribution.
        - Use line graph for displaying changes over time or continuous data or relationship between two continuous variables.
        - Use scatter plot for showing the relationship between two continuous variables, to identify any patterns and cluster of data.
        - Use table when presenting exact numerical data in organized rows and columns.

        The graph instruction MUST have the following structure in JSON format, selecting the relevant keys based on the visual type:
        {{
            "type": "bar" or "pie" or "line" or "scatter" or "table",
            "x_labels": ["label 1", "label 2", "label 3", "label 4"] for bar or line graphs,
            "x_values": [value 1, value 2, value 3, value 4] for scatter plot,
            "y_values": [value 1, value 2, value 3, value 4] for bar or line graphs,
            "labels": ["label 1", "label 2", "label 3", "label 4"] for pie chart,
            "sizes": [value 1, value 2, value 3, value 4] for pie chart,
            "y_label": "label for the y axis" for bar, line, scatter,
            "title": "title of the graph or table",
           "labels" : [ "label 1", "label 2", "label 3" ] for multiple lines in line graphs,
           "data": [
                    {{ "column1": "value1", "column2": "value2", ... }},
                    {{ "column1": "value3", "column2": "value4", ... }},
                    ...
                   ] for table
        }}

        Output the response in JSON format with the following structure:
        {{
          "questions" : [
            {{
                "question": "question text",
                "options": ["option a","option b", "option c", "option d"],
                "graph_instruction": {{"type": "bar" or "pie" or "line" or "scatter" or "table", ...}},
                "answer": "Correct answer of the question",
                "explanation": "Explanation of the question"
            }},
              {{
                "question": "question text",
                "options": ["option a","option b", "option c", "option d"],
                "graph_instruction": {{"type": "bar" or "pie" or "line" or "scatter" or "table", ...}},
                "answer": "Correct answer of the question",
                "explanation": "Explanation of the question"
            }}
           ]
        }}
"""

DATA_SOURCE_PROMPT_TEMPLATE = """
Analyze the following content and generate {num} {question_type} questions.

Source Type: {source_type}
Content Length: {content_length} characters

Guidelines for Question Generation:
1. Focus on key concepts and important information from the content
2. Ensure questions are directly based on the provided content
3. Vary the difficulty level of questions
4. Avoid redundant or overlapping questions
5. Create clear and unambiguous questions
6. Include relevant context where necessary

Content:
{topic}

{base_template}

Additional Context:
{custom_instructions}

Requirements:
1. Generate exactly {num} questions
2. Follow the specified question type format: {question_type}
3. Ensure all questions are factually accurate
4. Provide clear explanations for answers

{format_instructions}
"""

class QnAEngine:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        if llm_config is None:
            llm_config = LLMConfig()
        self.llm = self._initialize_llm(llm_config)
        self.pdf_loader = PdfFileLoader()
        self.url_loader = UrlLoader()
        self.embeddings = None

    def _initialize_llm(self, llm_config: LLMConfig):
        if llm_config.custom_model:
            return llm_config.custom_model
        else:
            return ChatOpenAI(
                model=llm_config.model_name,
                api_key=llm_config.api_key,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                base_url=llm_config.base_url,
                default_headers=llm_config.default_headers
            )

    def _get_parser_and_model(self, question_type: QuestionType, response_model: Optional[Type[Any]] = None):
        if response_model:
            return PydanticOutputParser(pydantic_object=response_model), response_model
        if question_type == "Multiple Choice":
            return PydanticOutputParser(pydantic_object=MCQList), MCQList
        elif question_type == "Short Answer":
            return PydanticOutputParser(pydantic_object=ShortAnswerQuestionList), ShortAnswerQuestionList
        elif question_type == "True/False":
            return PydanticOutputParser(pydantic_object=TrueFalseQuestionList), TrueFalseQuestionList
        elif question_type == "Fill in the Blank":
            return PydanticOutputParser(pydantic_object=FillInBlankQuestionList), FillInBlankQuestionList
        elif response_model == VisualMCQList:
            return PydanticOutputParser(pydantic_object=VisualMCQList), VisualMCQList
        else:
            raise ValueError(f"Unsupported question type or response model: {question_type}, {response_model}")


    def _get_prompt_template(self, question_type: QuestionType, custom_template: Optional[str] = None):
        if custom_template == "graph":
            return VISUAL_QUESTION_PROMPT_TEMPLATE
        elif custom_template:
            return custom_template
        else:
            base_template = f"""
            Generate {{num}} {question_type} question(s) based on the given topic.
            Topic: {{topic}}

            For each question, provide:
            1. The question
            2. The correct answer
            3. An explanation (optional)
            """

            if question_type == "Multiple Choice":
                base_template += "\n4. A list of options (including the correct answer)"
            elif question_type == "Short Answer":
                base_template += "\n4. A list of relevant keywords"
            elif question_type == "True/False":
                base_template += "\n4. The correct answer as a boolean (true/false)"
            elif question_type == "Fill in the Blank":
                base_template += "\n4. The word or phrase to be filled in the blank"

            return base_template



    def _create_vector_store(self, content: str) -> Chroma:
        if self.embeddings is None:
            self.embeddings = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(content)
        vectorstore = Chroma.from_texts(texts, self.embeddings)
        return vectorstore

    def _setup_retrieval_qa(self, vector_store: Chroma) -> RetrievalQA:
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
        )
    def _load_data(self, source: str, source_type: str) -> str:
     
        if source_type == 'pdf':
            return self.pdf_loader.load_data(source)
        elif source_type == 'url':
            return self.url_loader.load_data(source)
        elif source_type == 'text':
            return source
        else:
            raise ValueError("Unsupported source type. Please use 'pdf', 'url', or 'text'.")
            
    def _handle_output_format(self, data: Any, output_format: Optional[OutputFormatType]) -> Union[Any, Tuple[Any, str]]:
        
        if output_format is None:
            return data
            
        formatter = OutputFormatter()
        if output_format == "pdf":
            output_file = formatter.to_pdf(data)
            return data, output_file
        elif output_format == "csv":
            output_file = formatter.to_csv(data)
            return data, output_file
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


    def _generate_and_save_visual(self, instruction, question_text, options, correct_answer):
        try:
            plt.figure(figsize=(10, 8))
            img_buffer = io.BytesIO()

            if instruction["type"] == "bar":
                plt.bar(instruction["x_labels"], instruction["y_values"], color="skyblue")
                plt.xlabel("Categories", fontsize=12)
                plt.ylabel(instruction["y_label"], fontsize=12)
                plt.title(instruction["title"], fontsize=14)
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.tight_layout()
                plt.savefig(img_buffer, format="png")

            elif instruction["type"] == "line":
                if isinstance(instruction["y_values"][0], list):
                    for i, y_vals in enumerate(instruction["y_values"]):
                        plt.plot(instruction["x_labels"], y_vals, marker="o", linestyle="-", label=instruction["labels"][i])
                else:
                    plt.plot(instruction["x_labels"], instruction["y_values"], marker="o", linestyle="-", color="b")

                plt.xlabel("X-axis", fontsize=12)
                plt.ylabel(instruction["y_label"], fontsize=12)
                plt.title(instruction["title"], fontsize=14)
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.legend()
                plt.tight_layout()
                plt.savefig(img_buffer, format="png")

            elif instruction["type"] == "pie":
                plt.pie(
                    instruction["sizes"],
                    labels=instruction["labels"],
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=plt.cm.Paired.colors
                )
                plt.title(instruction["title"], fontsize=14)
                plt.tight_layout()
                plt.savefig(img_buffer, format="png")

            elif instruction["type"] == "scatter":
                plt.scatter(instruction["x_values"], instruction["y_values"], color="r", alpha=0.7)
                plt.xlabel("X-axis", fontsize=12)
                plt.ylabel(instruction["y_label"], fontsize=12)
                plt.title(instruction["title"], fontsize=14)
                plt.grid(axis="both", linestyle="--", alpha=0.7)
                plt.tight_layout()
                plt.savefig(img_buffer, format="png")

            elif instruction["type"] == "table":
                df = pd.DataFrame(instruction["data"])
                img_buffer = io.BytesIO()
                dfi.export(df, img_buffer, table_conversion="matplotlib")

            plt.close()
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

            if instruction["type"] != "table":
                display(HTML(f'<img src="data:image/png;base64,{img_base64}" style="max-width:500px; max-height:400px;">'))
            else:
                display(HTML(f'<img src="data:image/png;base64,{img_base64}" style="max-width:500px;">'))

            print("\nQuestion:", question_text)
            for idx, option in enumerate(options, start=1):
                print(f"{chr(64 + idx)}. {option}")
            print("Correct Answer:", correct_answer)
            print("-" * 80)

            return img_base64

        except Exception as e:
            print(f"Error generating visualization: {e}")
            return None


    def _display_visual_questions(self, ques: VisualMCQList):
        if ques and ques.questions:
            for q_data in ques.questions:
                instruction = q_data.graph_instruction
                question_text = q_data.question
                options = q_data.options
                correct_answer = q_data.answer

                self._generate_and_save_visual(instruction.dict(), question_text, options, correct_answer)
                print(q_data)
        else:
            print("Failed to generate visual questions or no questions were returned.")


    def generate_visual_questions(
        self,
        topic: str,
        num: int = 1,
        custom_instructions: Optional[str] = None,
        output_format: Optional[OutputFormatType] = None,
        **kwargs
    ) -> Optional[VisualMCQList]:
        parser, model = self._get_parser_and_model("Multiple Choice", VisualMCQList)
        format_instructions = parser.get_format_instructions()
        template = self._get_prompt_template("Multiple Choice", "graph")

        if custom_instructions:
            template += f"\n\nAdditional Instructions:\n{custom_instructions}"

        template += "\n\nThe response should be in JSON format.\n{format_instructions}"

        question_prompt = PromptTemplate(
            input_variables=["num", "topic"],
            template=template,
            partial_variables={"format_instructions": format_instructions}
        )

        question_chain = question_prompt | self.llm
        results = question_chain.invoke(
            {"num": num, "topic": topic, **kwargs},
        )
        results = results.content

        try:
            structured_output = parser.parse(results)

            if output_format:
                self._handle_output_format(structured_output, output_format)

            if isinstance(structured_output, VisualMCQList):
                self._display_visual_questions(structured_output)

            return structured_output
        except Exception as e:
            print(f"Error parsing output in generate_visual_questions: {e}")
            print("Raw output:")
            print(results)
            return None


    def generate_questions(
        self,
        topic: str,
        num: int = 1,
        question_type: QuestionType = "Multiple Choice",
        prompt_template: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        response_model: Optional[Type[Any]] = None,
        output_format: Optional[OutputFormatType] = None,
        **kwargs
    ) -> Any:
        parser, model = self._get_parser_and_model(question_type, response_model)
        format_instructions = parser.get_format_instructions()
        template = self._get_prompt_template(question_type, prompt_template)

        if custom_instructions:
            template += f"\n\nAdditional Instructions:\n{custom_instructions}"

        template += "\n\nThe response should be in JSON format.\n{format_instructions}"

        question_prompt = PromptTemplate(
            input_variables=["num", "topic"],
            template=template,
            partial_variables={"format_instructions": format_instructions}
        )

        question_chain = question_prompt | self.llm
        results = question_chain.invoke(
            {"num": num, "topic": topic, **kwargs},
        )
        results = results.content

        try:
            structured_output = parser.parse(results)

            if output_format:
                self._handle_output_format(structured_output, output_format)


            return structured_output
        except Exception as e:
            print(f"Error parsing output in generate_questions: {e}")
            print("Raw output:")
            return model()


    def generate_questions_from_data(
        self,
        source: str,
        source_type: str,
        num: int,
        question_type: QuestionType = "Multiple Choice",
        prompt_template: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        response_model: Optional[Type[Any]] = None,
        output_format: Optional[OutputFormatType] = None,
        max_content_length: int = 10000,
        **kwargs
    ) -> Any:
        """
        Generate questions from various data sources with improved handling and validation.
        """
        if not source:
            raise ValueError("Source cannot be empty")
        if num <= 0:
            raise ValueError("Number of questions must be positive")
            
        try:
            # Load and preprocess content
            content = self._load_data(source, source_type)
            
            if not content or len(content.strip()) == 0:
                raise ValueError(f"No content loaded from {source_type} source")
                
            # Truncate content if too long
            content_length = len(content)
            if content_length > max_content_length:
                logging.warning(f"Content length ({content_length}) exceeds maximum ({max_content_length}). Truncating...")
                content = content[:max_content_length]
                content_length = max_content_length

            # Set default response model to DataSourceMCQList for Multiple Choice questions
            if question_type == "Multiple Choice" and response_model is None:
                response_model = DataSourceMCQList

            # Get parser and base template
            parser, model = self._get_parser_and_model(question_type, response_model)
            base_template = self._get_prompt_template(question_type, prompt_template)
            format_instructions = parser.get_format_instructions()

            # Use the data source specific prompt template
            source_prompt = PromptTemplate(
                template=DATA_SOURCE_PROMPT_TEMPLATE,
                input_variables=[
                    "num",
                    "question_type",
                    "source_type",
                    "content_length",
                    "topic",
                    "base_template",
                    "custom_instructions",
                    "format_instructions"
                ]
            )

            # Format the prompt with all necessary information
            formatted_prompt = source_prompt.format(
                num=num,
                question_type=question_type,
                source_type=source_type,
                content_length=content_length,
                topic=content,
                base_template=base_template if base_template else "",
                custom_instructions=custom_instructions if custom_instructions else "No additional instructions provided.",
                format_instructions=format_instructions
            )

            # Generate questions using the formatted prompt
            start_time = time.time()
            result = self.generate_questions(
                topic=formatted_prompt,
                num=num,
                question_type=question_type,
                response_model=response_model,
                output_format=output_format,
                **kwargs
            )

            # Add source metadata and processing stats for DataSourceMCQList
            if isinstance(result, DataSourceMCQList):
                result.source_metadata = {
                    "source_type": source_type,
                    "content_length": content_length,
                    "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                
                result.processing_stats = {
                    "processing_time": f"{time.time() - start_time:.2f}s",
                    "num_questions_generated": len(result.questions),
                }

                # Add source information to each question
                for question in result.questions:
                    question.source_type = source_type
                    question.source_location = getattr(source, 'name', str(source))
                    # Extract relevant content segment (e.g., the sentence or paragraph containing the answer)
                    question.content_segment = self._extract_relevant_segment(content, question.answer)
            
            if not result or (hasattr(result, 'questions') and len(result.questions) == 0):
                logging.warning("No questions were generated")
                
            return result
            
        except Exception as e:
            logging.error(f"Error in generate_questions_from_data: {str(e)}")
            if response_model:
                return response_model(questions=[])
            else:
                model = self._get_parser_and_model(question_type)[1]
                return model(questions=[])

    def _extract_relevant_segment(self, content: str, answer: str, context_chars: int = 200) -> str:
        """
        Extract relevant content segment around the answer.
        
        Args:
            content: Full content text
            answer: Answer to look for
            context_chars: Number of characters of context to include before and after
        
        Returns:
            str: Relevant content segment
        """
        try:
            # Find the position of the answer in the content
            answer_pos = content.lower().find(answer.lower())
            if answer_pos == -1:
                return ""
            
            # Calculate start and end positions for the segment
            start = max(0, answer_pos - context_chars)
            end = min(len(content), answer_pos + len(answer) + context_chars)
            
            # Extract the segment
            segment = content[start:end].strip()
            
            # Add ellipsis if we're not at the start/end of the content
            if start > 0:
                segment = "..." + segment
            if end < len(content):
                segment = segment + "..."
                
            return segment
            
        except Exception as e:
            logging.warning(f"Error extracting relevant segment: {str(e)}")
            return ""

    def generate_questions_with_rag(
        self,
        source: str,
        source_type: str,
        num: int,
        question_type: QuestionType = "Multiple Choice",
        prompt_template: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        response_model: Optional[Type[Any]] = None,
        learning_objective: str = "",
        difficulty_level: str = "",
        output_format: Optional[OutputFormatType] = None,
        **kwargs
    ) -> Any:
       
        # Initialize embeddings if not already done
        if self.embeddings is None:
            self.embeddings = OpenAIEmbeddings()
            
        # Load content using the existing loader method
        content = self._load_data(source, source_type)
        
        # Create vector store using the existing method
        vector_store = self._create_vector_store(content)
        
        # Set up retrieval QA chain using the existing method
        qa_chain = self._setup_retrieval_qa(vector_store)
        
        # Get appropriate parser and model based on question type
        parser, model = self._get_parser_and_model(question_type, response_model)
        format_instructions = parser.get_format_instructions()
        
        # Get base prompt template if not provided
        if prompt_template is None:
            prompt_template = self._get_prompt_template(question_type)
        
        # Construct full prompt with learning objectives and difficulty
        full_prompt = f"""
        Generate {num} {question_type} questions based on the following content:
        
        {{topic}}
        
        Learning Objective: {{learning_objective}}
        Difficulty Level: {{difficulty_level}}
        
        Ensure that the questions:
        1. Are directly relevant to the learning objective
        2. Match the specified difficulty level
        3. Test understanding rather than mere recall
        4. Are clear, unambiguous, and grammatically correct
        5. For multiple choice questions, include plausible distractors
        
        {{format_instructions}}
        """
        
        # Add custom instructions if provided
        if custom_instructions:
            full_prompt += f"\n\nAdditional Instructions:\n{{custom_instructions}}"
        
        # Create prompt template
        question_prompt = PromptTemplate(
            input_variables=["topic", "learning_objective", "difficulty_level", "custom_instructions"],
            template=full_prompt,
            partial_variables={"format_instructions": format_instructions}
        )
        
        # Use the TextSplitter to chunk content if it's too large
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(content)
        
        all_results = []
        total_questions = 0
        
        # Process chunks until we have enough questions
        for i, chunk in enumerate(chunks):
            if total_questions >= num:
                break
                
            # Calculate how many questions to request from this chunk
            questions_needed = min(num - total_questions, max(1, num // len(chunks) + 1))
            
            # Format the prompt with current chunk
            query = question_prompt.format(
                topic=chunk,
                learning_objective=learning_objective,
                difficulty_level=difficulty_level,
                custom_instructions=custom_instructions or "",
                **kwargs
            )
            
            # Retrieve context and generate questions
            try:
                retrieval_results = qa_chain.invoke({
                    "query": query, 
                    "n_results": min(3, len(chunks))  # Adjust based on content size
                })
                
                # Parse the results
                chunk_results = parser.parse(retrieval_results["result"])
                
                # Extract questions from the result
                if hasattr(chunk_results, "questions"):
                    questions = chunk_results.questions
                elif isinstance(chunk_results, list):
                    questions = chunk_results
                else:
                    questions = [chunk_results]
                
                all_results.extend(questions)
                total_questions += len(questions)
                
            except Exception as e:
                # Add import for logger if not already present
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error processing chunk {i}: {str(e)}")
                # Continue with next chunk instead of failing
                continue
        
        # Limit to requested number of questions
        final_results = all_results[:num]
        
        # Create the structured output
        structured_output = model(questions=final_results)
        
        # Handle output format if specified
        if output_format:
            return self._handle_output_format(structured_output, output_format)
            
        return structured_output


    def generate_similar_options(self, question, correct_answer, num_options=3):
        llm = self.llm
        prompt = f"Generate {num_options} incorrect but plausible options similar to this correct answer: {correct_answer} for this question: {question}. Provide only the options, separated by semicolons. The options should not precede or end with any symbols, it should be similar to the correct answer."
        response = llm.predict(prompt)
        return response.split(';')

    def _process_math_result(self, math_result: Any) -> str:
        if isinstance(math_result, dict):
            if 'answer' in math_result:
                return math_result['answer'].split('Answer: ')[-1].strip()
            elif 'result' in math_result:
                return math_result['result'].strip()

        result_str = str(math_result)
        if 'Answer:' in result_str:
            return result_str.split('Answer:')[-1].strip()

        lines = result_str.split('\n')
        for line in reversed(lines):
            if line.strip().replace('.', '').isdigit():
                return line.strip()

        raise ValueError("Could not extract numerical result from LLMMathChain response")

    def generate_mcq_math(
        self,
        topic: str,
        num: int = 1,
        question_type: QuestionType = "Multiple Choice",
        prompt_template: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        response_model: Optional[Type[Any]] = None,
        **kwargs
    ) -> Any:
        if response_model is None:
            parser = PydanticOutputParser(pydantic_object=MCQListMath)
        else:
            parser = PydanticOutputParser(pydantic_object=response_model)

        format_instructions = parser.get_format_instructions()

        template = self._get_prompt_template(question_type, prompt_template)


        prompt_template = """
            You are an Academic AI assistant specialized in generating multiple-choice math questions.
            Generate {num} multiple-choice questions (MCQ) based on the given topic.
            Each question MUST be a mathematical computation question.

            For each question:
            1. Make sure it requires mathematical calculation
            2. Set requires_math to true
            3. Provide clear numerical values
            4. Ensure the question has a single, unambiguous answer

            Topic: {topic}

            Format each question to include:
            - A clear mathematical problem
            - Four distinct numerical options
            - The correct answer
            - A step-by-step explanation
            """

        if custom_instructions:
            prompt_template += f"\n\nAdditional Instructions:\n{custom_instructions}"

        prompt_template += "\nThe response should be in JSON format.\n{format_instructions}"

        question_prompt = PromptTemplate(
            input_variables=["num", "topic"],
            template=prompt_template,
            partial_variables={"format_instructions": format_instructions}
        )

        question_chain = question_prompt | self.llm
        results = question_chain.invoke(
            {"num": num, "topic": topic, **kwargs},
        )
        results = results.content

        try:
            structured_output = parser.parse(results)
        except Exception as e:
            print(f"Error parsing output: {e}")
            print("Raw output:")
            return MCQListMath()

        llm_math = LLMMathChain.from_llm(llm=self.llm, verbose=True)

        for question in structured_output.questions:
            if question.requires_math:
                try:
                    math_result = llm_math.invoke({"question": question.question})

                    try:
                        solution = self._process_math_result(math_result)

                        numerical_solution = float(solution)
                        formatted_solution = f"{numerical_solution:.2f}"

                        question.explanation += f"\n\nMath solution: {formatted_solution}"

                        correct_option = Option(text=formatted_solution, correct='true')

                        variations = [0.9, 1.1, 1.2]
                        incorrect_options = []

                        for var in variations:
                            wrong_val = numerical_solution * var
                            incorrect_options.append(
                                Option(
                                    text=f"{wrong_val:.2f}",
                                    correct='false'
                                )
                            )

                        question.options = [correct_option] + incorrect_options
                        random.shuffle(question.options)

                    except (ValueError, TypeError) as e:
                        print(f"Error processing numerical result: {e}")
                        raise

                except Exception as e:
                    print(f"LLMMathChain failed to answer: {str(e)}")
                    question.explanation += "\n\nMath solution: Unable to compute."
                    question.options = [
                        Option(text="Unable to compute", correct='true'),
                        Option(text="N/A", correct='false'),
                        Option(text="N/A", correct='false'),
                        Option(text="N/A", correct='false')
                    ]

        return structured_output

    def _extract_video_id(self, url: str) -> str:
        pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?(?:embed\/)?(?:v\/)?(?:shorts\/)?(?:live\/)?(?:feature=player_embedded&v=)?(?:e\/)?(?:\/)?([^\s&amp;?#]+)'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        raise ValueError("Invalid YouTube URL")

    def _get_youtube_transcript(self, video_id: str, target_language: str = 'en') -> tuple[str, str]:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            available_languages = [transcript.language_code for transcript in transcript_list]

            if not available_languages:
                raise ValueError("No transcripts available for this video")

            try:
                transcript = transcript_list.find_transcript([target_language])
                return TextFormatter().format_transcript(transcript.fetch()), target_language
            except:
                transcript = transcript_list.find_transcript(available_languages)
                original_language = transcript.language_code

                if transcript.is_translatable and target_language != original_language:
                    translated = transcript.translate(target_language)
                    return TextFormatter().format_transcript(translated.fetch()), target_language

                return TextFormatter().format_transcript(transcript.fetch()), original_language

        except Exception as e:
            error_message = str(e).lower()
            if "transcriptsdisabled" in error_message:
                raise ValueError(
                    "This video does not have subtitles/closed captions enabled. "
                    "Available languages: " + ", ".join(available_languages)
                )
            elif "notranscriptfound" in error_message:
                raise ValueError(
                    f"No transcript found for language '{target_language}'. "
                    f"Available languages: {', '.join(available_languages)}"
                )
            else:
                raise ValueError(f"Error fetching transcript: {str(e)}")

    def generate_questions_from_youtube(
        self,
        url: str,
        num: int = 1,
        question_type: QuestionType = "Multiple Choice",
        prompt_template: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        response_model: Optional[Type[Any]] = None,
        output_format: Optional[OutputFormatType] = None,
        target_language: str = 'en',
        preserve_original_language: bool = False,
        **kwargs
    ) -> Any:
        try:
            video_id = self._extract_video_id(url)
            transcript, detected_language = self._get_youtube_transcript(video_id, target_language)

            if not transcript:
                raise ValueError("No transcript content retrieved from the video")

            language_context = f"\nContent language: {detected_language}"
            if detected_language != target_language and not preserve_original_language:
                language_context += f"\nGenerate questions in {target_language}"

            video_context = f"\nThis content is from a YouTube video (ID: {video_id}). {language_context}"
            if custom_instructions:
                custom_instructions = video_context + "\n" + custom_instructions
            else:
                custom_instructions = video_context

            return self.generate_questions_from_data(
                source=transcript,
                source_type="text",
                num=num,
                question_type=question_type,
                prompt_template=prompt_template,
                custom_instructions=custom_instructions,
                response_model=response_model,
                output_format=output_format,
                target_language=target_language,
                **kwargs
            )

        except ValueError as ve:
            raise ValueError(f"YouTube processing error: {str(ve)}")
        except Exception as e:
            raise Exception(f"Unexpected error processing YouTube video: {str(e)}")

    def _load_image(self, source: str) -> str:
        try:
            if source.startswith(('http://', 'https://')):
                return source
            elif source.startswith('data:image'):
                return source
            else:
                image = Image.open(source)
                if image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
                
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"

        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")

    def solve_doubt(
        self,
        image_source: str,
        prompt: str = "Explain how to solve this problem",
        custom_instructions: Optional[str] = None,
        detail_level: Literal["low", "medium", "high"] = "medium",
        focus_areas: Optional[List[str]] = None,
        **kwargs
    ) -> SolvedDoubt:
        """
        Analyze an image and provide detailed explanation with solution steps.
        
        Args:
            image_source: Path or URL to the image
            prompt: Custom prompt for analysis
            custom_instructions: Additional instructions for analysis
            detail_level: Level of detail in explanation
            focus_areas: Specific aspects to focus on
            **kwargs: Additional parameters for the model
        
        Returns:
            SolvedDoubt: Object containing explanation, steps, and additional notes
        """
        if not image_source:
            raise ValueError("Image source (path or URL) is required")

        try:
            image_content = self._load_image(image_source)
            
            # Create parser for structured output
            parser = PydanticOutputParser(pydantic_object=SolvedDoubt)
            format_instructions = parser.get_format_instructions()

            # Construct the prompt with all parameters
            base_prompt = f"Analyze the image and {prompt}\n"
            if focus_areas:
                base_prompt += f"\nFocus on these aspects: {', '.join(focus_areas)}"
            base_prompt += f"\nProvide a {detail_level}-detail explanation"
            
            system_message = SystemMessage(
                content="You are a helpful assistant that responds in Markdown. Help with math homework."
            )

            human_message_content = f"""
            {base_prompt}
            
            Provide:
            1. A detailed explanation
            2. Step-by-step solution (if applicable)
            3. Any additional notes or tips
            
            {custom_instructions or ''}
            
            {format_instructions}
            """

            human_message = HumanMessage(content=[
                {"type": "text", "text": human_message_content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_content,
                        "detail": "high" if detail_level == "high" else "low"
                    }
                }
            ])

            response = self.llm.invoke(
                [system_message, human_message],
                **kwargs
            )

            try:
                return parser.parse(response.content)
            except Exception as e:
                # Fallback if parsing fails
                return SolvedDoubt(
                    explanation=response.content,
                    steps=[],
                    additional_notes="Note: Response format was not structured as requested."
                )

        except Exception as e:
            error_msg = f"Error in solve_doubt: {type(e).__name__}: {str(e)}"
            print(error_msg)
            return SolvedDoubt(
                explanation=error_msg,
                steps=[],
                additional_notes="An error occurred during processing."
            )

   
