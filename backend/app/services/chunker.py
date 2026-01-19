from typing import List

class TextChunker:
    @staticmethod
    def chunk_text(text: str, strategy: str = "paragraph") -> List[str]:
        """
        Splits text into chunks based on the selected strategy.
        """
        if not text:
            return []
            
        text = text.strip()
        
        if strategy == "paragraph":
            # Simple split by double newlines or single newlines
            # normalizing line endings
            normalized_text = text.replace("\r\n", "\n")
            paragraphs = [p.strip() for p in normalized_text.split("\n") if p.strip()]
            return paragraphs
            
        # Fallback or Todo: 'sentence' or 'punctuation' strategy
        # For now, default to paragraph split
        return [text]
