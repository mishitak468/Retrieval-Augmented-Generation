"""
Main RAG System - Interactive Question Answering
Orchestrates the complete Retrieval-Augmented Generation pipeline
"""

import sys
import os
from encode import QueryEncoder
from vector_db import VectorDatabase
from llm_generation import LLMGenerator


class RAGSystem:
    def __init__(self, model_path, model_type='tinyllama', top_k=3):
        """
        Initialize the complete RAG system

        Args:
            model_path: Path to LLM model file
            model_type: Type of LLM model
            top_k: Number of documents to retrieve
        """
        print("="*70)
        print("Initializing RAG System".center(70))
        print("="*70)

        self.top_k = top_k

        # Component 1: Query Encoder
        print("\n[1/3] Loading Query Encoder...")
        self.encoder = QueryEncoder()

        # Component 2 & 3: Vector Database & Document Retrieval
        print("\n[2/3] Loading Vector Database...")
        self.vector_db = VectorDatabase()
        self.vector_db.load_documents('preprocessed_documents.json')
        self.vector_db.build_index(index_type='flat')

        # Component 5: LLM Generator
        print("\n[3/3] Loading Language Model...")
        self.llm = LLMGenerator(model_path, model_type=model_type)

        print("\n" + "="*70)
        print("✓ RAG System Ready!".center(70))
        print("="*70)

    def answer_question(self, query, verbose=True):
        """
        Execute the complete RAG pipeline for a user query

        Args:
            query: User's natural language question
            verbose: If True, print intermediate steps

        Returns:
            Generated answer string
        """
        if verbose:
            print("\n" + "="*70)
            print(f"Query: {query}")
            print("="*70)

        try:
            # Step 1: Encode query
            if verbose:
                print("\n[Step 1] Encoding query...")
            query_embedding = self.encoder.encode(query)

            # Step 2: Vector search
            if verbose:
                print("[Step 2] Searching vector database...")
            retrieved_docs = self.vector_db.search(
                query_embedding,
                k=self.top_k,
                remove_duplicates=True
            )

            # Step 3: Display retrieved documents
            if verbose:
                print(
                    f"[Step 3] Retrieved top {len(retrieved_docs)} documents:")
                for i, (doc_id, distance, text) in enumerate(retrieved_docs, 1):
                    print(
                        f"\n  Document {i} (ID: {doc_id}, Distance: {distance:.2f}):")
                    print(f"  {text[:200]}...")

            # Step 4: Create augmented prompt
            if verbose:
                print("\n[Step 4] Creating augmented prompt...")
            augmented_prompt = self.llm.create_augmented_prompt(
                query, retrieved_docs)

            # Step 5: Generate response
            if verbose:
                print("[Step 5] Generating response with LLM...")
                print("-"*70)

            response = self.llm.generate(augmented_prompt, max_tokens=256)

            return response

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different question.")
            return None

    def run_interactive(self):
        """
        Run interactive command-line interface
        """
        print("\n" + "="*70)
        print("Interactive RAG System".center(70))
        print("="*70)
        print("\nType your questions below. Commands:")
        print("  - Type 'quit' or 'exit' to stop")
        print("  - Type 'help' for usage tips")
        print("="*70)

        while True:
            try:
                # Get user input
                query = input("\n🤔 Your Question: ").strip()

                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for using the RAG system. Goodbye!")
                    break

                if query.lower() == 'help':
                    self._print_help()
                    continue

                if not query:
                    print("⚠️  Please enter a question.")
                    continue

                # Process query through RAG pipeline
                answer = self.answer_question(query, verbose=True)

                # Display answer
                print("\n" + "="*70)
                print("🤖 Answer:")
                print("="*70)
                print(answer)
                print("="*70)

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again with a different question.")

    def _print_help(self):
        """Print help information"""
        print("\n" + "="*70)
        print("RAG System Help".center(70))
        print("="*70)
        print("""
How it works:
  1. You ask a question
  2. System encodes your question into a vector
  3. System searches for relevant documents
  4. System augments your question with retrieved context
  5. LLM generates an answer based on the context

Tips for better results:
  • Ask specific questions
  • Questions about topics in the MS MARCO dataset work best
  • The system retrieves top 3 most relevant documents
  • Responses are grounded in retrieved documents

Example questions:
  • "What causes animals to lose fur?"
  • "How do I improve my computer's performance?"
  • "What are the symptoms of a cold?"
        """)
        print("="*70)


def main():
    """
    Main entry point for the RAG system
    """
    # Configuration
    MODEL_CONFIGS = {
        'tinyllama': {
            'path': 'tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf',
            'type': 'tinyllama',
            'url': 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf'
        },
        'qwen2-1.5b': {
            'path': 'qwen2-1_5b-instruct-q4_0.gguf',
            'type': 'qwen2-1.5b',
            'url': 'https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_0.gguf'
        },
        'llama3.2': {
            'path': 'Llama-3.2-3B-Instruct-Q4_K_M.gguf',
            'type': 'llama3.2',
            'url': 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf'
        }
    }

    # Check for preprocessed documents
    if not os.path.exists('preprocessed_documents.json'):
        print("❌ Error: preprocessed_documents.json not found!")
        print("Please run data_preprocess.py first to create this file.")
        sys.exit(1)

    # Find available model
    model_path = None
    model_type = None

    for name, config in MODEL_CONFIGS.items():
        if os.path.exists(config['path']):
            model_path = config['path']
            model_type = config['type']
            print(f"✓ Found model: {config['path']}")
            break

    if model_path is None:
        print("❌ Error: No LLM model found!")
        print("\nPlease download one of these models:")
        for name, config in MODEL_CONFIGS.items():
            print(f"\n{name}:")
            print(f"  wget {config['url']}")
        sys.exit(1)

    # Initialize RAG system
    try:
        rag = RAGSystem(model_path, model_type=model_type, top_k=3)

        # Run interactive mode
        rag.run_interactive()

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
