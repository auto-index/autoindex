import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor
import re


class Preprocessor(BasePreprocessor):
    name = "jurisdiction_context_amplification"
    description = "Add jurisdiction context chunks with state name repetition to improve state-specific retrieval"

    # Common state names and abbreviations to detect and amplify
    STATES = {
        'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware',
        'florida', 'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky',
        'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi',
        'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey', 'new mexico',
        'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania',
        'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas', 'utah', 'vermont',
        'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming', 'puerto rico', 'district of columbia'
    }
    
    STATE_ABBREVIATIONS = {
        'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia', 'ks',
        'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny',
        'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv',
        'wi', 'wy', 'pr', 'dc'
    }

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        chunks = []
        
        for doc in docs:
            # Always keep the original document as the primary chunk
            original_chunk = Chunk(
                chunk_id=f"{doc.doc_id}_0",
                doc_id=doc.doc_id,
                text=doc.text
            )
            chunks.append(original_chunk)
            
            # Detect jurisdiction context from the document text
            detected_states = self._detect_states(doc.text)
            
            # If we found state references, create an enhanced context chunk
            if detected_states:
                # Create a jurisdiction-focused chunk that amplifies state references
                jurisdiction_text = self._create_jurisdiction_chunk(doc.text, detected_states)
                
                jurisdiction_chunk = Chunk(
                    chunk_id=f"{doc.doc_id}_jurisdiction",
                    doc_id=doc.doc_id,
                    text=jurisdiction_text
                )
                chunks.append(jurisdiction_chunk)
        
        return chunks

    def _detect_states(self, text: str) -> List[str]:
        """Detect state names and abbreviations in the text."""
        detected = set()
        text_lower = text.lower()
        
        # Check for full state names
        for state in self.STATES:
            if state in text_lower:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(state) + r'\b'
                if re.search(pattern, text_lower):
                    detected.add(state.title())  # Normalize capitalization
        
        # Check for state abbreviations (more restrictive to avoid false positives)
        # Only consider abbreviations that appear in common legal contexts
        words = re.findall(r'\b[A-Z]{2}\b', text)  # Find potential abbreviations
        for word in words:
            if word.lower() in self.STATE_ABBREVIATIONS:
                # Try to confirm this is actually a state reference by context
                # Check if it appears near words like "state", "law", etc.
                context_pattern = rf'(state|statute|law|code|title).*?{re.escape(word)}|{re.escape(word)}.*?(state|statute|law|code|title)'
                if re.search(context_pattern, text, re.IGNORECASE):
                    detected.add(word)
        
        return list(detected)

    def _create_jurisdiction_chunk(self, original_text: str, states: List[str]) -> str:
        """Create a chunk that emphasizes jurisdiction context."""
        # Create a context prefix that repeats state information
        state_mentions = []
        for state in states:
            # Repeat the state name several times to increase TF
            state_mentions.append(f"jurisdiction: {state} state law: {state} legal code: {state}")
        
        # Join all state mentions
        jurisdiction_context = " ".join(state_mentions)
        
        # Add jurisdiction markers throughout the text to strengthen BM25 signal
        # We insert these markers at the beginning and periodically through the text
        sentences = re.split(r'(?<=[.!?])\s+', original_text)
        enhanced_sentences = []
        
        # Add jurisdiction context at the beginning
        enhanced_sentences.append(jurisdiction_context)
        
        # Periodically insert jurisdiction markers throughout the text
        for i, sentence in enumerate(sentences):
            enhanced_sentences.append(sentence)
            # Every 10 sentences, reinsert the jurisdiction markers
            if (i + 1) % 10 == 0 and i < len(sentences) - 1:
                enhanced_sentences.append(jurisdiction_context)
        
        return " ".join(enhanced_sentences)
