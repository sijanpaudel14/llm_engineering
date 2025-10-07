import base64
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError
from IPython.display import Image, display, Audio, Markdown
import mimetypes
import os

# Assuming key_utils.py exists and works
from key_utils import get_next_key

import langchain_google_genai.chat_models as chat_mod
class LLMHandler:
    # --- Model Constants ---
    # Using modern, flexible models that support conversation history is recommended
    TEXT_MODEL = "gemini-2.0-flash"
    AUDIO_MODEL = "gemini-2.5-flash-preview-tts"
    IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation"
    def __init__(self, system_message="You are a helpful assistant."):
        self.conversation_history = []
        self.system_message = system_message
        self._original_chat_with_retry = chat_mod._chat_with_retry

 # --- Patch retry ---
    def _patch_retry(self, enable_retry: False):
        """Enable or disable internal retry inside LangChain/Google API"""
        if not enable_retry:
            def no_retry_chat_with_retry(**kwargs):
                generation_method = kwargs.pop("generation_method")
                metadata = kwargs.pop("metadata", None)
                return generation_method(
                    request=kwargs.get("request"),
                    retry=None,
                    timeout=None,
                    metadata=metadata
                )
            chat_mod._chat_with_retry = no_retry_chat_with_retry
        else:
            chat_mod._chat_with_retry = self._original_chat_with_retry
        
        # --- Helper: extract base64 image ---
    def _get_image_base64(self, response):
        for block in response.content:
            if isinstance(block, dict) and "image_url" in block:
                return block["image_url"]["url"].split(",")[-1]
        return None

    

    # --- REFACTORED AND FIXED: Message Building ---
    def _build_messages(self, user_input, task_type="text"):
        """
        Correctly builds the message list for the API, preserving history for all task types.
        """
        messages = []

        # FIX: Only add system message for conversational tasks that support it.
        if task_type in ["text", "tool"]:
            messages.append(SystemMessage(content=self.system_message))

        # FIX: Convert the entire history correctly for the model.
        for h in self.conversation_history:
            role = h.get("role")
            content = h.get("content")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        
        # Always add the latest user input.
        messages.append(HumanMessage(content=user_input))
        return messages

    # --- REFACTORED AND FIXED: History Updating ---
    def _update_history(self, user_input, assistant_response_content):
        """
        Correctly updates the internal history with the latest turn.
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        # FIX: Always store the actual content from the assistant.
        self.conversation_history.append({"role": "assistant", "content": assistant_response_content})
    
    def clear_history(self):
        """Helper to reset the conversation."""
        self.conversation_history = []
        print("Conversation history cleared.")


    def _get_raw_stream_generator(self, user_input, llm, messages):
        """A private helper to create the base generator that includes history updates."""
        full_response_content = ""
        try:
            for chunk in llm.stream(messages):
                full_response_content += chunk.content
                yield chunk  # Yield the original LangChain message chunk
        finally:
            # This block executes when the generator is exhausted or closed.
            self._update_history(user_input, full_response_content)

        
            
    def stream_and_process(self, user_input, on_chunk_callback=None, **kwargs):
        """
        Streams a response, displays it in real-time as Markdown, and
        optionally processes each chunk with a user-provided callback function.

        This method handles the entire streaming lifecycle in one call.

        Args:
            user_input (str): The user's prompt.
            on_chunk_callback (callable, optional): A function to call on each
                response chunk. The function should accept one argument: the chunk.
            **kwargs: Additional arguments for the run() method.

        Returns:
            str: The final, complete response text.
        """
        # 1. Get the raw generator from the run method.
        kwargs['stream'] = True
        raw_generator = self.run(user_input, **kwargs)

        if not hasattr(raw_generator, '__iter__'):
             print("❌ Streaming failed. Could not get a generator.")
             return ""

        # 2. Set up the display and start the process. This all happens immediately.
        print("Assistant is responding...")
        display_handle = display(Markdown(""), display_id=True)
        full_response_content = ""

        # 3. Consume the generator internally, driving the entire process.
        try:
            for chunk in raw_generator:
                full_response_content += chunk.content
                
                # Action 1: Update the live Markdown display.
                display_handle.update(Markdown(f"""{full_response_content}"""))
                
                # Action 2: Execute the user's callback function if provided.
                if on_chunk_callback:
                    try:
                        on_chunk_callback(chunk)
                    except Exception as e:
                        print(f"🚨 Error in your callback function: {e}")
        finally:
            print("\n--- Stream finished ---")

        # 4. Return the final, assembled text.
        return full_response_content

    # --- THE UNIFIED RUN METHOD ---
    def run(self, user_input, task_type="text", tools=None, max_retries=11, enable_retry=False,width=400,height=None, stream=False, display_text=True,max_tokens = None):
        """
        A single, unified method to handle text, audio, image, and tool generation.
        This method RETURNS data instead of displaying it.
        
        Args:
            user_input (str or list): The user's prompt or message.
            task_type (str): The type of task ("text", "audio", "image", etc.).
            tools (list, optional): A list of tools for the model to use. Defaults to None.
            max_retries (int): The maximum number of times to retry with a new API key.
            enable_retry (bool): Whether to enable the internal LangChain retry mechanism.
            width (int): The display width for generated images.
            height (int): The display height for generated images.
            stream (bool): If True for text generation, yields the response in chunks.
            
        Returns:
            dict: A dictionary containing the result for non-streaming tasks.
            Generator[str]: If stream=True for a text task, yields response content in chunks.
        """
        self._patch_retry(enable_retry)
        
                # 1. Select model
        model_name = {
            "text": self.TEXT_MODEL,
            "tool": self.TEXT_MODEL,
            "audio": self.AUDIO_MODEL,
            "image": self.IMAGE_MODEL,
            "audio_transcription": self.TEXT_MODEL,
            "video_transcription": self.TEXT_MODEL,
            "image_analysis": self.TEXT_MODEL, #Use the multimodal model for transcription
        }.get(task_type, self.TEXT_MODEL)

        # 2. Build messages (with a special case for transcription)
        if task_type in ["audio_transcription", "video_transcription",  "image_analysis"]:
            # For transcription, the user_input is already a fully formed HumanMessage
            messages = [user_input]
        else:
            messages = self._build_messages(user_input, task_type=task_type)
        
        # 3. Centralized Retry Loop
        for attempt in range(max_retries):
            api_key, user_name = get_next_key()
            print(f"➡️ Attempt {attempt + 1}/{max_retries} using key from '{user_name}'...")
            quota_errors = ["quota", "exceed", "overloaded", "exhausted", "limit"]
            try:
                llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key,max_tokens=max_tokens)
                
                # 4. Handle different task types
                if task_type == "audio":
                    tts_message = [HumanMessage(content=user_input)] # Use only the direct input for TTS
                    response = llm.invoke(tts_message, generation_config={"response_modalities": ["AUDIO"]})
                    audio_bytes = response.additional_kwargs.get("audio")
                    self._update_history(user_input, "[Generated audio in response to prompt]")
                    return {"type": "audio", "data": audio_bytes, "text": response.content}

                elif task_type == "image":
                    # For images, we just need the user prompt
                    response = llm.invoke(messages, generation_config={"response_modalities": ["TEXT","IMAGE"]})
                    image_base64 = self._get_image_base64(response)
                    if image_base64:
                        display(Image(data=base64.b64decode(image_base64), width=width, height=height))
                        # Save only prompt for reference
                        self._update_history(user_input, "[Generated image in response to prompt]")
                        print( "image_generated")
                        return {"type": "image", "data": image_base64, "text": "Image generated successfully."}
                    else:
                        print("No image returned")

                else: # Handles "text" and "tool" and "audio_transcription"
                    model_to_invoke = llm
                    if tools:
                        model_to_invoke = llm.bind_tools(tools, tool_choice="any")
                        
                    if stream and task_type == 'text' and not tools:
                        return self._get_raw_stream_generator(user_input, llm, messages)
                        
                    response = model_to_invoke.invoke(messages)

                    # Tool Execution Logic
                    if response.tool_calls:
                        messages.append(response)
                        for tool_call in response.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            # Find the callable tool function
                            matched_tool_func = next((t for t in tools if getattr(t, '__name__', None) == tool_name), None)
                            if matched_tool_func:
                                result = matched_tool_func(**tool_args)
                                print(f"✅ [TOOL] Called '{tool_name}' with {tool_args}. Result: {result}")
                            else:
                                result = f"Error: Tool '{tool_name}' not found."
                                print(f"❌ [TOOL] {result}")
                            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
                        
                        # Call the model again with the tool results
                        response = llm.invoke(messages)
                     # UPDATED: Differentiated history logging
                    if task_type == ["audio_transcription","video_transcription"]:
                        prompt_summary = f"[Transcription requested for an media file]"
                        self._update_history(prompt_summary, response.content)
                    elif task_type == "image_analysis":
                         # Extract the text part of the prompt for a better history log
                        prompt_text = next((part['text'] for part in user_input.content if isinstance(part, dict) and part['type'] == 'text'), 'Analyze image')
                        prompt_summary = f"[{prompt_text}]"
                        self._update_history(prompt_summary, response.content)
                    else:
                        self._update_history(user_input, response.content)
                        

                return {"type": "text", "data": response.content, "text": response.content}
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {e.__class__.__name__} - {e}")
                error_msg = str(e).lower()
                # check if any keyword from list is inside error message
                if any(err in error_msg for err in quota_errors):
                    print("⚠️ Quota/limit issue, trying next key...")
                    continue   # try next key
                else:
                    raise   # if it's another kind of error, stop
            
            # If loop finishes with no success
        raise RuntimeError("All API keys failed or quota exceeded after all attempts.")

    #******************************************************************#
    # --- NEW: GENERATE AUDIO FROM TEXT METHOD ---
    #******************************************************************#
    def generate_audio(self, user_input, play_audio=True):
        """
        A convenience method that generates audio and optionally plays it.
        This method acts as a wrapper around the main `run()` method.
        """
        print(f"--- Generating Audio for: '{user_input}' ---")
        response_dict = self.run(user_input, task_type="audio")
        
        if play_audio and response_dict and response_dict.get('type') == 'audio' and response_dict.get('data'):
            print("Audio generation successful. Playing audio...")
            try:
                audio_bytes = response_dict['data']
                audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")
                play(audio_segment)
                print("Playback finished.")
            except Exception as e:
                print(f"❌ Error playing audio: {e}")
        elif not response_dict or not response_dict.get('data'):
            print("❌ Audio generation failed or no audio data was returned.")
            return None
            
        return response_dict
        
    #******************************************************************#
    # --- NEW: GENERATE TEXT FROM AUDIO METHOD ---
    #******************************************************************#
    def generate_text_from_audio(self, file_name, enable_retry=False):
        """
        Transcribes an audio file into text using a multimodal model.

        Args:
            file_name (str): The path to the audio file (e.g., .mp3, .wav, .flac).
            enable_retry (bool): Whether to enable the built-in LangChain retry mechanism.

        Returns:
            str: The transcribed text from the audio, or None if an error occurred.
        """
        print(f"--- Transcribing Audio from: '{file_name}' ---")

        if not os.path.exists(file_name):
            print(f"❌ Error: Audio file not found at '{file_name}'")
            return None

        # 1. Automatically detect MIME type
        mime_type, _ = mimetypes.guess_type(file_name)
        if not mime_type or not mime_type.startswith("audio"):
            print(f"❌ Error: Could not determine a valid audio MIME type for '{file_name}'")
            return None
        
        print(f"Detected MIME type: {mime_type}")

        # 2. Read file and encode in base64
        with open(file_name, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")

        # 3. Construct the special multimodal message
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Transcribe the audio and provide the full text.",
                },
                {
                    "type": "media",
                    "data": encoded_audio,
                    "mime_type": mime_type,
                },
            ]
        )

        # 4. Call the main run method with the new task type
        response_dict = self.run(
            user_input=message,
            task_type="audio_transcription",
            enable_retry=enable_retry
        )

        if response_dict and response_dict.get('type') == 'text':
            return response_dict['text']
        else:
            print("❌ Audio transcription failed or no text was returned.")
            return None

    #******************************************************************#
    # --- NEW: GENERATE TEXT FROM VIDEO METHOD ---
    #******************************************************************#
    def generate_transcript_from_video(self, file_name, enable_retry=False):
        """
        Transcribes the audio track from a video file into text.

        Args:
            file_name (str): The path to the video file (e.g., .mp4, .mov, .webm).
            enable_retry (bool): Whether to enable the built-in LangChain retry mechanism.

        Returns:
            str: The transcribed text from the video's audio, or None if an error occurred.
        """
        print(f"--- Transcribing Video from: '{file_name}' ---")

        if not os.path.exists(file_name):
            print(f"❌ Error: Video file not found at '{file_name}'")
            return None

        # 1. Automatically detect MIME type for the video
        mime_type, _ = mimetypes.guess_type(file_name)
        if not mime_type or not mime_type.startswith("video"):
            print(f"❌ Error: Could not determine a valid video MIME type for '{file_name}'")
            return None
        
        print(f"Detected MIME type: {mime_type}")

        # 2. Read file and encode in base64
        with open(file_name, "rb") as video_file:
            encoded_video = base64.b64encode(video_file.read()).decode("utf-8")

        # 3. Construct the special multimodal message for video transcription
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Please transcribe the audio from this video. Provide only the spoken words as text.",
                },
                {
                    "type": "media",
                    "data": encoded_video,
                    "mime_type": mime_type,
                },
            ]
        )

        # 4. Call the main run method with the new task type
        response_dict = self.run(
            user_input=message,
            task_type="video_transcription",
            enable_retry=enable_retry
        )

        if response_dict and response_dict.get('type') == 'text':
            return response_dict['text']
        else:
            print("❌ Video transcription failed or no text was returned.")
            return None

    #******************************************************************#
    # --- NEW: GENERATE TEXT FROM IMAGE METHOD ---
    #******************************************************************#
    def generate_text_from_image(self, image_source, prompt="Describe this image in detail.", enable_retry=False):
        """
        Analyzes an image and generates a textual description or answer.

        Args:
            image_source (str): The path to a local image file OR a public URL to an image.
            prompt (str): The question or command for the model regarding the image.
            enable_retry (bool): Whether to enable the built-in LangChain retry mechanism.

        Returns:
            str: The text generated by the model, or None if an error occurred.
        """
        print(f"--- Analyzing Image from: '{image_source[:70]}...' ---")
        image_url_content = ""

        # 1. Check if the source is a URL or a local file
        if image_source.startswith("http://") or image_source.startswith("https://"):
            image_url_content = image_source
        elif os.path.exists(image_source):
            # It's a local file, so we need to encode it
            mime_type, _ = mimetypes.guess_type(image_source)
            if not mime_type or not mime_type.startswith("image"):
                print(f"❌ Error: Could not determine a valid image MIME type for '{image_source}'")
                return None
            
            with open(image_source, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            image_url_content = f"data:{mime_type};base64,{encoded_image}"
        else:
            print(f"❌ Error: Image file not found at '{image_source}'")
            return None

        # 2. Construct the special multimodal message
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url_content},
            ]
        )
        
        # 3. Call the main run method with the new task type
        response_dict = self.run(
            user_input=message,
            task_type="image_analysis",
            enable_retry=enable_retry
        )

        if response_dict and response_dict.get('type') == 'text':
            return response_dict
        else:
            print("❌ Image analysis failed or no text was returned.")
            return None


import os
import time
from huggingface_hub import InferenceClient
from langchain_core.runnables import Runnable
from IPython.display import Video, display

class VideoGenerator(Runnable):
    def __init__(self, model, api_key, filename_prefix="video", provider = "replicate"):
        self.client = InferenceClient(provider= provider, token=api_key)
        self.model = model
        self.filename_prefix = filename_prefix

    def _generate_filename(self):
        timestamp = int(time.time())
        return f"{self.filename_prefix}_{timestamp}.mp4"

    def invoke(self, prompt, config=None):
        """Generate video from raw prompt"""
        filename = self._generate_filename()
        video = self.client.text_to_video(prompt, model=self.model)
        with open(filename, "wb") as f:
            f.write(video)
        return filename

    def play(self, filepath):
        """Display video in notebook"""
        display(Video(filepath, embed=True))


# USE case 
# --- Usage ---
# api_token = os.getenv("HF_TOKEN")

# video_gen = VideoGenerator(
#     model="genmo/mochi-1-preview",
#     api_key=api_token,
#     provider="fal-ai"
# )

# # Generate and save video
# file_path = video_gen.invoke("A cinematic video of a spaceship landing on Mars at sunset")
# print(f"Video saved as {file_path}")

# # Play video in notebook
# video_gen.play(file_path)