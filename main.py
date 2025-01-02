import json
import logging
from modules.init import load_deepfilter_model
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

model, df_state = None, None


def save_audio_file(data, title):
    # Save the audio file locally
    with open(f'{title}.wav', 'wb') as f:
        f.write(data)
        logger.info('saved!!!')

def lambda_handler(event, context):
    # logger.info(f"Event received: {json.dumps(event)}")
    logger.info("00000000000")
    audio_name = ''

    # Check if the event contains body and is base64 encoded
    if 'body' in event and event.get('isBase64Encoded', False):
        import base64
        # Decode the base64 body
        audio_data = base64.b64decode(event['body'])
        audio_name = event['audio_filename']
        save_audio_file(audio_data, audio_name)        

    # logger.info(event)
    # audio_file = event.get('audio_file')
    
    # logger.info(audio_file)
    
    global model, df_state
    
    try:
        # Load model if not already loaded
        if model is None or df_state is None:
            model, df_state = load_deepfilter_model(model, df_state)
        
        logger.info("DeepFilter model loaded successfully")
        
        # audio_path = download_file(
        #     "https://github.com/Rikorose/DeepFilterNet/raw/e031053/assets/noisy_snr0.wav",
        #     download_dir=".",
        # )
        # audio, _ = load_audio(audio_path, sr=df_state.sr())

        audio, _ = load_audio(f'{audio_name}.wav', sr=df_state.sr())

        # Denoise the audio
        enhanced = enhance(model, df_state, audio)

        # Save for listening
        save_audio(f"{audio_name}_enhanced.wav", enhanced, df_state.sr())

        # Read the enhanced audio file and encode it to base64
        with open(f"{audio_name}_enhanced.wav", 'rb') as f:
            enhanced_audio_data = f.read()

        # Base64 encode the enhanced audio to send it as part of the response
        encoded_audio = base64.b64encode(enhanced_audio_data).decode('utf-8')

        # Return the base64-encoded enhanced audio in the response
        return {
            'statusCode': 200,
            'isBase64Encoded': True,
            'headers': {
                'Content-Type': 'audio/wav',
                'Content-Disposition': f'attachment; filename="{audio_name}_enhanced.wav"'
            },
            'body': encoded_audio,
            'message': 'Successfully enhanced the audio.'
        }
    except Exception as e:
        logger.error(f"Error processing the audio file: {e}")
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal Server Error'})
        }

        # return {
        #     'success': True,
        #     'message': 'DeepFilter model loaded successfully'
        # }
        
    # except Exception as e:
    #     logger.error(f"Error loading DeepFilter model: {str(e)}", exc_info=True)
    #     return {
    #         'success': False,
    #         'message': f'Error loading DeepFilter model: {str(e)}'
    #     }