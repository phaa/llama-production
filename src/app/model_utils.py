import torch
import io
import os  
from google.cloud import vision
from PIL import Image
from model_loader import get_model_and_processor


def preprocess_conversation(processor, model, conversation, images):
    """
    Essa função encapsula a lógica para ler e processar as conversas
    """
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=images, text=prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=1000)

    # Extrair apenas a resposta, descartando o prompt
    input_length = inputs.input_ids.shape[1]  # Número de tokens no prompt
    result = processor.decode(output[0][input_length:], skip_special_tokens=True)

    return result.strip()


async def check_vegetation(images: list[Image.Image]) -> str:
    processor, model = get_model_and_processor()

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
                        Analise estas imagens do mesmo poste elétrico. 
                        Existe vegetação encostando nos postes, fios ou transformador? 
                        Responda apenas com SIM ou NÃO
                    """,
                },
            ],
        }
    ]

    for image in images:
        conversation[0]["content"].append({"type": "image"})

    result = preprocess_conversation(processor, model, conversation, images)
    return result


async def recognize_pole_switch(images: list[Image.Image]) -> str:
    processor, model = get_model_and_processor()

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
                        Analise estas imagens do mesmo poste elétrico.
                        Qual o tipo de chave seccionadora instalada? 
                        Responda se é 'Fusível' ou 'Fusível Religadora' ou 'Faca' ou 'A gás'. Responda apenas o nome da chave. Caso não encontre nenhuma, responda apenas null.
                    """,
                },
            ],
        }
    ]

    for image in images:
        conversation[0]["content"].append({"type": "image"})

    result = preprocess_conversation(processor, model, conversation, images)
    return result


def recognize_pole_transformer(image: Image.Image) -> str:
    # Verifique se a variável de ambiente está configurada (opcional, mas útil para debugging)
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        print(f"GOOGLE_APPLICATION_CREDENTIALS está definida como: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
    else:
        print("A variável de ambiente GOOGLE_APPLICATION_CREDENTIALS não está definida.")

    client = vision.ImageAnnotatorClient()  # A biblioteca cliente usará as credenciais da variável de ambiente

    # Converter o objeto PIL.Image.Image para bytes
    buffered = io.BytesIO()
    image_2 = image.copy()
    image_2.save(buffered, format="JPEG")  # Você pode escolher outro formato se preferir
    image_bytes = buffered.getvalue()

    image_vision = vision.Image(content=image_bytes)

    response = client.text_detection(image=image_vision)
    texts = response.text_annotations

    texto_detectado = ""
    if texts:
        texto_detectado = texts[0].description if texts[0].description else ""

    processor, model = get_model_and_processor()

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
                        Esta imagem contém a placa de identificação de um transformador. Utilize a saída de um OCR abaixo para preencher o seguinte formulário. Retorne a saída em formato JSON, onde as chaves são os nomes dos campos do formulário e os valores são as informações correspondentes encontradas na placa.
                        Note que as primeiras linhas do OCR são os dados do fabricante.
                        Os campos do formulário são:

                        Transformador de distribuição:
                        {
                        "Código": "",
                        "Matrícula": "",
                        "Poste": "",
                        "Autoprotegido": "",
                        "Blindado": "",
                        "Capacidade da chave": "",
                        "Capacidade do ELO": "",
                        "Exclusivo para IP": "",
                        "Fases": "",
                        "Indica Paralelo": "",
                        "Indica Rede MEN": "",
                        "Potência nominal (kVA)": "",
                        "Med. Balanço Energ.": "",
                        "Posto": "",
                        "TAP": "",
                        "Zona de Distribuição Aérea": "",
                        "Tipo de aterramento": "",
                        "Tipo de instalação": "",
                        "Tipo de ligação": "",
                        "Alimentador": "",
                        "Tipo de montagem": "",
                        "Tipo de transformador": ""
                        }

                        Equipamento do transformador:
                        {
                        "Cód. Trafo": "",
                        "Cód. Equipamento": "",
                        "Classe": "",
                        "Data de fabricação": "",
                        "Fabricante": "",
                        "Número de tombamento": "",
                        "Matrícula (Equipamento)": "",
                        "Meio de isolação": "",
                        "Potência nominal (kVA) (Equipamento)": "",
                        "Quantidade de fases": "",
                        "Série": "",
                        "Tensão primária (KV)": "",
                        "Tensão secundária (KV)": "",
                        "Tipo de autoproteção": "",
                        "Tipo de ligação (Equipamento)": ""
                        }
                    """ 
                    + texto_detectado
                    + "Preencha os valores correspondentes em cada campo encontrados na placa da imagem. Tente raciocinar e fazer as aproximações entre as informaçãos. Se uma informação não estiver presente, deixe o valor como null, voce não é obrigado a responder todos os campos do formulário",
                },
                {"type": "image"}
            ],
        }
    ]

    result: str = preprocess_conversation(processor, model, conversation, images=image)
    result = result.replace("\n", "").replace("\\", "")

    return result