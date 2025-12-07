"""
PIPELINE AVEC SAUVEGARDE AUTOMATIQUE
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import torch
import gc
import json
import shutil
from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

print("="*70)
print("PIPELINE VLM + LLM + GraphRAG AVEC SAUVEGARDE")
print("="*70)

# Nettoyage GPU initial
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   Mémoire totale: {total_mem:.2f} GB")

# ============================================
# CLASSE VLM
# ============================================

class Florence2VLM:
    """Vision Language Model - Florence-2-base"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n🔄 Loading Florence-2-base...")
        print(f"   Device: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base",
            trust_remote_code=True
        )

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(0) / 1024**3
            print(f"✅ VLM chargé - Mémoire GPU: {mem:.2f} GB")

    def analyze(self, image_path):
        """Analyse une image avec OCR"""
        print(f"   📸 Analyzing {image_path}...")

        img = Image.open(image_path).convert('RGB')

        inputs = self.processor(
            text="<OCR_WITH_REGION>",
            images=img,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device, dtype=torch.float16) if v.dtype == torch.float32 else v.to(self.device) 
                  for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                num_beams=3,
                use_cache=False
            )

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result = text.replace("<OCR_WITH_REGION>", "").strip()

        return result

# ============================================
# CLASSE LLM
# ============================================

class Qwen25WithGraphRAG:
    """LLM avec GraphRAG"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n🔄 Loading Qwen2.5-3B-Instruct...")
        print(f"   Device: {self.device}")
        print(f"   Mode: FP16")

        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            trust_remote_code=True
        )

        self._setup_graphrag()

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(0) / 1024**3
            print(f"✅ LLM chargé - Mémoire GPU: {mem:.2f} GB")

    def _setup_graphrag(self):
        """Configure GraphRAG"""
        print("\n📚 Setting up GraphRAG...")

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            self.vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings,
                collection_name="fertility_collection"
            )

            self.graphrag_enabled = True
            print("✅ GraphRAG loaded from chroma_db")

        except Exception as e:
            print(f"⚠️ GraphRAG setup failed: {e}")
            self.graphrag_enabled = False

    def generate(self, query, vlm_result):
        """Génère une réponse avec contexte GraphRAG"""

        context = ""
        if self.graphrag_enabled:
            print("   🔍 Retrieving context from GraphRAG...")
            try:
                docs = self.vector_store.similarity_search(query, k=3)
                context = "\n\n".join([doc.page_content[:400] for doc in docs])
                print(f"   ✅ Retrieved {len(docs)} documents")
            except Exception as e:
                print(f"   ⚠️ Retrieval failed: {e}")
                context = "No context available"
        else:
            context = "No GraphRAG context available"

        prompt = f"""You are a compassionate fertility assistant.

MEDICAL CONTEXT FROM KNOWLEDGE BASE:
{context}

IMAGE ANALYSIS (Hormone Panel OCR):
{vlm_result}

INSTRUCTIONS:
- Ground your response in the provided medical context
- Be warm, empathetic, and clear
- Acknowledge any uncertainty
- ALWAYS include a medical disclaimer
- Never provide definitive medical diagnosis
- Encourage consultation with healthcare professionals

USER QUERY: {query}

RESPONSE:"""

        print("   💭 Generating response...")

        messages = [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

# ============================================
# FONCTION DE SAUVEGARDE
# ============================================

def save_models(vlm, llm, save_dir="./saved_models"):
    """Sauvegarde les modèles"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Sauvegarde dans: {save_dir}")
    
    # Sauvegarder VLM
    print("   📸 Sauvegarde VLM...")
    vlm_dir = save_dir / "florence2"
    vlm_dir.mkdir(exist_ok=True)
    vlm.model.save_pretrained(vlm_dir / "model")
    vlm.processor.save_pretrained(vlm_dir / "processor")
    print("      ✅ VLM sauvegardé")
    
    # Sauvegarder LLM
    print("   🤖 Sauvegarde LLM...")
    llm_dir = save_dir / "qwen"
    llm_dir.mkdir(exist_ok=True)
    llm.model.save_pretrained(llm_dir / "model")
    llm.tokenizer.save_pretrained(llm_dir / "tokenizer")
    print("      ✅ LLM sauvegardé")
    
    # Copier GraphRAG
    if Path("./chroma_db").exists():
        print("   📚 Copie GraphRAG...")
        graphrag_dir = save_dir / "chroma_db"
        if graphrag_dir.exists():
            shutil.rmtree(graphrag_dir)
        shutil.copytree("./chroma_db", graphrag_dir)
        print("      ✅ GraphRAG sauvegardé")
    
    # Metadata
    info = {
        "vlm_model": "microsoft/Florence-2-base",
        "llm_model": "Qwen/Qwen2.5-3B-Instruct",
        "has_graphrag": Path("./chroma_db").exists()
    }
    
    with open(save_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✅ Sauvegarde terminée: {save_dir.absolute()}")

# ============================================
# PIPELINE PRINCIPAL
# ============================================

def run_pipeline_and_save():
    """Execute pipeline et sauvegarde"""
    
    print("\n" + "="*70)
    print("ÉTAPE 1: ANALYSE VLM")
    print("="*70)

    vlm = Florence2VLM()
    vlm_result = vlm.analyze("hormone_panel.png")

    print("\n📊 Résultat VLM:")
    print("-"*70)
    print(vlm_result)
    print("-"*70)

    with open("vlm_result.txt", "w") as f:
        f.write(vlm_result)
    print("💾 Sauvegardé: vlm_result.txt")

    print("\n" + "="*70)
    print("ÉTAPE 2: CHARGEMENT LLM")
    print("="*70)

    llm = Qwen25WithGraphRAG()

    print("\n" + "="*70)
    print("ÉTAPE 3: SAUVEGARDE DES MODÈLES")
    print("="*70)
    
    # SAUVEGARDER AVANT DE LIBÉRER!
    save_models(vlm, llm, "./saved_models")

    # Maintenant on peut libérer VLM
    print("\n🧹 Libération mémoire VLM...")
    del vlm
    torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "="*70)
    print("ÉTAPE 4: GÉNÉRATION")
    print("="*70)

    query = "Based on these hormone levels, what should I know?"
    print(f"\n❓ Question: {query}")

    response = llm.generate(query, vlm_result)

    print("\n💬 Réponse:")
    print("-"*70)
    print(response)
    print("-"*70)

    if torch.cuda.is_available():
        final_mem = torch.cuda.memory_allocated(0) / 1024**3
        print(f"\n📊 Mémoire GPU finale: {final_mem:.2f} GB")

    print("\n" + "="*70)
    print("✅ PIPELINE TERMINÉ - MODÈLES SAUVEGARDÉS!")
    print("="*70)
    print("\n📦 Pour télécharger:")
    print("!zip -r saved_models.zip saved_models/")
    print("from google.colab import files")
    print("files.download('saved_models.zip')")

if __name__ == "__main__":
    run_pipeline_and_save()
