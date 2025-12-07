"""
Interface SIMPLE - Florence2 + Qwen + GraphRAG
THÈME FERTILITÉ ROSE/FLORAL 🌸
"""

import gradio as gr
import torch
import gc
from pathlib import Path
from PIL import Image

from stt import SpeechToText
from tts import TextToSpeech


class SimpleMedicalAssistant:
    """Version optimisée - Thème Fertilité"""
    
    def __init__(self, models_dir="/content/drive/MyDrive/fertility_models/saved_models"):
        print("🚀 Initialisation...")
        self.models_dir = Path(models_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Modules légers (toujours en mémoire)
        print("  📝 STT...")
        self.stt = SpeechToText(model_type="whisper", model_size="base", language="en", device="cpu")
        
        print("  🔊 TTS...")
        self.tts = TextToSpeech(backend="gtts", language="en", output_dir="./audio_responses")
        
        print("  📚 GraphRAG...")
        self._load_graphrag()
        
        print("\n✅ Prêt! (Florence2 et Qwen se chargent à la demande)")
    
    def _load_graphrag(self):
        """Charge GraphRAG"""
        try:
            chroma_path = self.models_dir / "chroma_db"
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            self.vector_store = Chroma(
                persist_directory=str(chroma_path),
                embedding_function=embeddings,
                collection_name="fertility_collection"
            )
            print(f"     ✅ GraphRAG chargé")
        except Exception as e:
            print(f"     ⚠️ GraphRAG erreur: {e}")
            self.vector_store = None
    
    def analyze_image_with_florence(self, image_path):
        """Charge Florence2 → Analyse → Libère"""
        print("\n🔄 Chargement Florence2...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            vlm_path = self.models_dir / "florence2"
            
            model = AutoModelForCausalLM.from_pretrained(
                vlm_path / "model",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="eager"
            ).to(self.device)
            
            processor = AutoProcessor.from_pretrained(
                vlm_path / "processor",
                trust_remote_code=True
            )
            
            print("✅ Florence2 chargé")
            
            img = Image.open(image_path).convert('RGB')
            print(f"✅ Image chargée: {img.size}")
            
            inputs = processor(
                text="<OCR_WITH_REGION>", 
                images=img, 
                return_tensors="pt"
            )
            
            if inputs is None:
                raise ValueError("Le processor a retourné None")
            
            if 'pixel_values' not in inputs:
                raise ValueError("pixel_values manquant dans inputs")
                
            if inputs['pixel_values'] is None:
                raise ValueError("pixel_values est None")
            
            print(f"✅ Inputs préparés: {list(inputs.keys())}")
            
            inputs = {
                k: v.to(self.device, dtype=torch.float16) if v.dtype == torch.float32 else v.to(self.device) 
                for k, v in inputs.items()
            }
            
            print("🔄 Analyse de l'image...")
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                    use_cache=False
                )
            
            if generated_ids is None:
                raise ValueError("model.generate() a retourné None")
            
            if not isinstance(generated_ids, torch.Tensor):
                raise ValueError(f"generated_ids n'est pas un Tensor, type: {type(generated_ids)}")
            
            if generated_ids.numel() == 0:
                raise ValueError("generated_ids est un tensor vide")
            
            print(f"✅ Génération OK, shape: {generated_ids.shape}")
            
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            result = text.replace("<OCR_WITH_REGION>", "").strip()
            
            if not result:
                result = "⚠️ Aucun texte détecté dans l'image"
            
            print(f"✅ Analyse terminée: {len(result)} caractères")
            
            print("🧹 Nettoyage mémoire...")
            del model, processor, inputs, generated_ids, img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return result
            
        except Exception as e:
            print(f"❌ ERREUR: {str(e)}")
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return f"❌ Impossible d'analyser l'image: {str(e)}"
    
    def generate_with_qwen(self, query, image_context=""):
        """Charge Qwen → Génère → Libère"""
        print("\n🔄 Chargement Qwen...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            llm_path = self.models_dir / "qwen"
            
            model = AutoModelForCausalLM.from_pretrained(
                llm_path / "model",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(self.device)
            
            tokenizer = AutoTokenizer.from_pretrained(
                llm_path / "tokenizer",
                trust_remote_code=True
            )
            
            print("✅ Qwen chargé, génération en cours...")
            
            context = ""
            if self.vector_store:
                try:
                    docs = self.vector_store.similarity_search(query, k=3)
                    context = "\n\n".join([doc.page_content[:400] for doc in docs])
                except:
                    context = "Context unavailable"
            
            prompt = f"""You are a compassionate fertility assistant.

MEDICAL CONTEXT:
{context}

{"IMAGE ANALYSIS:" if image_context else ""}
{image_context}

INSTRUCTIONS:
- Be warm, empathetic, and clear
- Ground response in medical context
- Include disclaimer at end
- Never give definitive diagnosis

USER QUERY: {query}

RESPONSE:"""
            
            messages = [
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            print("🧹 Libération Qwen...")
            del model, tokenizer, inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return response.strip()
            
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return f"❌ Erreur Qwen: {str(e)}"
    
    def process_query(self, text_input, audio_input, image_input):
        """Handler principal"""
        
        question = ""
        image_context = ""
        
        if audio_input is not None:
            print("\n📢 Mode: Audio")
            stt_result = self.stt.transcribe_file(audio_input)
            if stt_result["success"]:
                question = stt_result["text"]
            else:
                return f"❌ Erreur STT: {stt_result['error']}", None
        
        elif text_input and text_input.strip():
            print("\n📝 Mode: Texte")
            question = text_input.strip()
        
        else:
            return "❌ Veuillez entrer une question (texte ou audio)", None
        
        if image_input is not None:
            print("\n📸 Détection d'image, analyse...")
            image_context = self.analyze_image_with_florence(image_input)
        
        print(f"\n💭 Question: {question}")
        answer = self.generate_with_qwen(question, image_context)
        
        result_text = f"""## 📝 Question:
> {question}

{"## 📸 Image analysée:" if image_context else ""}
{"```" + image_context[:200] + "...```" if image_context else ""}

## 🤖 Réponse:

{answer}

---
*⚕️ Disclaimer: Cette information est éducative. Consultez toujours un professionnel de santé.*
"""
        
        print("\n🔊 Génération audio...")
        tts_result = self.tts.synthesize(answer)
        audio_output = tts_result["output_file"] if tts_result["success"] else None
        
        return result_text, audio_output
    
    def create_interface(self):
        """Interface THÈME FERTILITÉ ROSE 🌸"""
        
        with gr.Blocks(
            title="Assistant Fertilité 🌸", 
            theme=gr.themes.Soft(
                primary_hue="pink",
                secondary_hue="rose",
            ),
            css="""
                .header-title {
                    text-align: center;
                    margin-bottom: 15px;
                    background: linear-gradient(135deg, #ffc0cb 0%, #ffb6c1 100%);
                    padding: 25px;
                    border-radius: 20px;
                    box-shadow: 0 4px 8px rgba(255, 182, 193, 0.4);
                }
                .input-section {
                    background: linear-gradient(135deg, #ffb6c1 0%, #ffc0cb 50%, #ffb3d9 100%);
                    padding: 15px;
                    border-radius: 15px;
                    margin-top: 10px;
                    box-shadow: 0 4px 6px rgba(255, 182, 193, 0.3);
                }
                .chat-container {
                    max-height: 350px;
                    overflow-y: auto;
                    padding: 15px;
                    border: 2px solid #ffb6c1;
                    border-radius: 15px;
                    background: linear-gradient(to bottom, #fff5f7 0%, #ffe4e9 100%);
                    margin-bottom: 15px;
                }
            """
        ) as demo:
            
            # HEADER ROSE
            with gr.Row():
                gr.Markdown(
                    """
# 🌸 Assistant Fertilité 🌸
### *Votre compagnon bienveillant pour votre parcours* 💕
                    """,
                    elem_classes="header-title"
                )
            
            # CONVERSATION
            gr.Markdown("### 🌺 Conversation")
            
            with gr.Group(elem_classes="chat-container"):
                chatbot = gr.Chatbot(
                    value=[],
                    label="",
                    height=150,  
                    show_label=False,
                    bubble_full_width=False
                )
            
            with gr.Row():
                output_audio = gr.Audio(
                    label="🎵 Écouter la réponse",
                    visible=True
                )
            
            # ZONE INPUT ROSE
            with gr.Group(elem_classes="input-section"):
                gr.Markdown("### 🌷 Posez votre question")
                
                with gr.Row():
                    text_input = gr.Textbox(
                        label="",
                        placeholder="💬 Écrivez votre question ici...",
                        lines=1,
                        max_lines=2,
                        scale=4,
                        container=False
                    )
                
                with gr.Row():
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="🎤",
                        scale=1,
                        container=False
                    )
                    
                    image_input = gr.Image(
                        type="filepath",
                        label="📋",
                        scale=1,
                        container=False,
                        height=100
                    )
                
                submit_btn = gr.Button(
                    "🌸 Envoyer",
                    variant="primary",
                    size="sm"
                )
            
            chat_history = gr.State([])
            
            def chat_interface(text_input, audio_input, image_input, history):
                result_text, audio_file = self.process_query(text_input, audio_input, image_input)
                
                if "## 📝 Question:" in result_text:
                    parts = result_text.split("## 🤖 Réponse:")
                    question_part = parts[0].replace("## 📝 Question:", "").strip()
                    question = question_part.split("\n")[0].replace(">", "").strip()
                    
                    if len(parts) > 1:
                        response = parts[1].split("---")[0].strip()
                    else:
                        response = "Erreur lors de la génération de la réponse."
                else:
                    question = text_input if text_input else "Question audio"
                    response = result_text
                
                history.append([question, response])
                
                return history, "", None, None, audio_file
            
            submit_btn.click(
                fn=chat_interface,
                inputs=[text_input, audio_input, image_input, chat_history],
                outputs=[chatbot, text_input, audio_input, image_input, output_audio]
            )
        
        return demo
