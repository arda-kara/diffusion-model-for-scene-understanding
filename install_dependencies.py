"""
Script to install required dependencies for the scene understanding system.
"""

import subprocess
import sys
import os

def install_spacy_model():
    """
    Install the required SpaCy model.
    """
    print("Installing SpaCy model 'en_core_web_sm'...")
    try:
        # First ensure spaCy is installed and up to date
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "spacy"])
        print("SpaCy installed/upgraded successfully.")
        
        # Install the English model
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("SpaCy model installed successfully.")
        
        # Verify the installation
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            print(f"SpaCy model loaded successfully with components: {', '.join(nlp.pipe_names)}")
            
            # Check if the parser component is available
            if 'parser' in nlp.pipe_names:
                print("Parser component is available.")
            else:
                print("Warning: Parser component is not available. Relationship extraction may be limited.")
                
            # Test a simple sentence
            doc = nlp("The cat sits on the mat.")
            print("Test sentence parsed successfully.")
            print("Sentence tokens:", [token.text for token in doc])
            print("Parts of speech:", [token.pos_ for token in doc])
            print("Dependencies:", [token.dep_ for token in doc])
            
        except Exception as e:
            print(f"Error verifying SpaCy model: {e}")
            print("Please try to reinstall it manually.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error installing SpaCy model: {e}")
        print("Please try to install it manually with:")
        print("python -m pip install --upgrade spacy")
        print("python -m spacy download en_core_web_sm")

def install_nltk_resources():
    """
    Install required NLTK resources.
    """
    print("Installing NLTK resources...")
    try:
        # First ensure nltk is installed and up to date
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "nltk"])
        print("NLTK installed/upgraded successfully.")
        
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        print("NLTK resources installed successfully.")
        
        # Verify the installation
        try:
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            
            # Test tokenization
            tokens = word_tokenize("This is a test sentence.")
            print("NLTK tokenization test successful:", tokens)
            
            # Test stopwords
            stops = stopwords.words('english')
            print(f"NLTK stopwords loaded successfully ({len(stops)} stopwords available).")
            
        except Exception as e:
            print(f"Error verifying NLTK resources: {e}")
            print("Please try to reinstall them manually.")
            
    except Exception as e:
        print(f"Error installing NLTK resources: {e}")
        print("Please try to install them manually with:")
        print("python -m pip install --upgrade nltk")
        print("python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"")

def main():
    """
    Main function.
    """
    print("Installing dependencies for the scene understanding system...")
    
    # Install SpaCy model
    install_spacy_model()
    
    # Install NLTK resources
    install_nltk_resources()
    
    print("\nAll dependencies installed. You can now use the scene understanding system.")
    print("\nIf you encounter any issues with NLP processing, try running the test script:")
    print("python test_nlp.py")

if __name__ == "__main__":
    main() 