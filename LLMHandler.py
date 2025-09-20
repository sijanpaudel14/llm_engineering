import base64
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError
from IPython.display import Image, display, Audio, Markdown

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


    # --- THE UNIFIED RUN METHOD ---
    def run(self, user_input, task_type="text", tools=None, max_retries=11, enable_retry=False,width=400,height=None):
        """
        A single, unified method to handle text, audio, image, and tool generation.
        This method RETURNS data instead of displaying it.
        """
        self._patch_retry(enable_retry)
        
        # 1. Select the correct model based on the task
        model_name = {
            "text": self.TEXT_MODEL,
            "tool": self.TEXT_MODEL,
            "audio": self.AUDIO_MODEL,
            "image": self.IMAGE_MODEL,
        }.get(task_type, self.TEXT_MODEL)

        # 2. Build the message list correctly
        messages = self._build_messages(user_input, task_type=task_type)
        
        # 3. Centralized Retry Loop
        for attempt in range(max_retries):
            api_key, user_name = get_next_key()
            print(f"➡️ Attempt {attempt + 1}/{max_retries} using key from '{user_name}'...")
            
            try:
                llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
                
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

                else: # Handles "text" and "tool"
                    model_to_invoke = llm
                    if tools:
                        model_to_invoke = llm.bind_tools(tools, tool_choice="any")
                    
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

                    self._update_history(user_input, response.content)
                    return {"type": "text", "data": response.content, "text": response.content}

            except (ResourceExhausted, GoogleAPICallError, ValueError) as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {e.__class__.__name__} - {e}")
                if attempt + 1 >= max_retries:
                    raise RuntimeError("All API keys failed or quota exceeded.") from e
                continue # Try the next key

        raise RuntimeError("All API keys failed or quota exceeded after all attempts.")

        # --- NEW CONVENIENCE METHOD FOR AUDIO ---
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


















# import base64
# from io import BytesIO
# from pydub import AudioSegment
# from pydub.playback import play
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
# from IPython.display import Image, display, Markdown
# from key_utils import get_next_key
# import langchain_google_genai.chat_models as chat_mod

# class LLMHandler:
#     # --- Model names ---
#     TEXT_MODEL = "gemini-2.0-flash"
#     AUDIO_MODEL = "gemini-2.5-flash-preview-tts"
#     IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation"

#     def __init__(self, system_message="You are a helpful assistant."):
#         self.conversation_history = []
#         self.system_message = system_message
#         self._original_chat_with_retry = chat_mod._chat_with_retry  # save original for patching

#     # --- Patch retry ---
#     def _patch_retry(self, enable_retry: False):
#         """Enable or disable internal retry inside LangChain/Google API"""
#         if not enable_retry:
#             def no_retry_chat_with_retry(**kwargs):
#                 generation_method = kwargs.pop("generation_method")
#                 metadata = kwargs.pop("metadata", None)
#                 return generation_method(
#                     request=kwargs.get("request"),
#                     retry=None,
#                     timeout=None,
#                     metadata=metadata
#                 )
#             chat_mod._chat_with_retry = no_retry_chat_with_retry
#         else:
#             chat_mod._chat_with_retry = self._original_chat_with_retry

#     # --- Helper: build messages ---
#     # --- Build messages for LLM ---
#     def _build_messages(self, user_input, task_type="text", system_override=None):
#         messages = []
#         if task_type in ["text", "tool"]:
#             messages.append(SystemMessage(content=system_override if system_override else self.system_message))
#         # Convert history
#         for h in self.conversation_history:
#             role = h.get("role")
#             content = h.get("content")
#             if role == "user":
#                 messages.append(HumanMessage(content=content))
#             elif role == "assistant":
#                 messages.append(AIMessage(content=content))
#         # Current input
#         if task_type in ["text", "tool"]:
#             messages.append(HumanMessage(content=user_input))
#         else:
#             messages = [HumanMessage(content=user_input)]
#         return messages

#     # --- Execute tools ---
#     def _execute_tools(self, messages, response):
#         """
#         Executes any tool calls returned by the model.
#         Updates messages with ToolMessage containing the result.
#         """
#         if tool_calls := getattr(response, "tool_calls", None):
#             messages.append(response)
#             for tool_call in tool_calls:
#                 tool_name = tool_call["name"]
#                 tool_args = tool_call["args"]

#                 # Find matching tool
#                 matched_tool = next((t for t in self.tools if t.name == tool_name), None)
#                 if matched_tool:
#                     result = matched_tool.func(**tool_args)
#                     print(f"[TOOL] Called {tool_name} with {tool_args}, returned: {result}")
#                 else:
#                     result = f"No implementation for {tool_name}"
#                     print(f"[TOOL] {result}")

#                 messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
#             return True
#         return False

#  # --- Update history ---
#     def _update_history(self, user_input, assistant_response, content_type="text"):
#         self.conversation_history.append({"role": "user", "content": user_input, "type": "text"})
#         # For images, store only the prompt, not actual image data
#         if content_type == "image":
#             self.conversation_history.append({"role": "assistant", "content": user_input, "type": "image"})
#         else:
#             self.conversation_history.append({"role": "assistant", "content": assistant_response, "type": content_type})


#     # --- Helper: extract base64 image ---
#     def _get_image_base64(self, response):
#         for block in response.content:
#             if isinstance(block, dict) and "image_url" in block:
#                 return block["image_url"]["url"].split(",")[-1]
#         return None

#     # --- TEXT GENERATION ---
#     def generate_text(self, user_input, model_name=None, enable_retry=False):
#         model_name = model_name or self.TEXT_MODEL
#         self._patch_retry(enable_retry)
#         messages = self._build_messages(user_input, task_type="text")
#         for attempt in range(11):
#             api_key, _ = get_next_key()
#             print(f"➡️ Using API key: {api_key[:10]} of {_} (Attempt {attempt+1})")
#             llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
#             try:
#                 response = llm.invoke(messages)
#                 text = ""
#                 if isinstance(response.content, str):
#                     text = response.content
#                 elif isinstance(response.content, list):
#                     for block in response.content:
#                         if isinstance(block, str):
#                             text += block
#                         elif isinstance(block, dict) and "text" in block:
#                             text += block["text"]
#                 self._update_history(user_input, text)
#                 display(Markdown(text))
#                 return text
#             except Exception as e:
#                 print(f"Attempt {attempt+1} failed: {e}")
#                 continue
#         raise RuntimeError("All API keys failed or quota exceeded.")

#     # --- STREAMING TEXT GENERATION ---
#     def generate_text_stream(self, user_input, model_name=None, enable_retry=False):
#         model_name = model_name or self.TEXT_MODEL
#         self._patch_retry(enable_retry)
#         messages = self._build_messages(user_input, task_type="text")
#         for attempt in range(11):
#             api_key, _ = get_next_key()
#             print(f"➡️ Using API key: {api_key[:10]} of {_} (Attempt {attempt+1})")
#             llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
#             try:
#                 display_handle = display(Markdown(""), display_id=True)
#                 response_text = ""
#                 for chunk in llm.stream(messages):
#                     if chunk.content:
#                         response_text += chunk.content
#                         display_handle.update(Markdown(response_text))
#                 self._update_history(user_input, response_text)
#                 return response_text
#             except Exception as e:
#                 print(f"Attempt {attempt+1} failed: {e}")
#                 continue
#         raise RuntimeError("All API keys failed or quota exceeded.")

#     # --- AUDIO GENERATION ---
#     def generate_audio(self, user_input, model_name=None, enable_retry=False):
#         model_name = model_name or self.AUDIO_MODEL
#         self._patch_retry(enable_retry)
#         for attempt in range(11):
#             api_key, _ = get_next_key()
#             print(f"➡️ Using API key: {api_key[:10]} of {_} (Attempt {attempt+1})")
#             llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
#             messages = self._build_messages(user_input, task_type="audio")
#             try:
#                 response = llm.invoke(messages, generation_config={"response_modalities": ["AUDIO"]})
#                 audio_bytes = response.additional_kwargs.get("audio")
#                 if audio_bytes:
#                     audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")
#                     play(audio_segment)
#                     self._update_history(user_input, "audio_played")
#                     return "audio_played"
#                 else:
#                     print("No audio returned")
#             except Exception as e:
#                 print(f"Attempt {attempt+1} failed: {e}")
#                 continue
#         raise RuntimeError("All API keys failed or quota exceeded.")

#     # --- IMAGE GENERATION ---
#     def generate_image(self, user_input, model_name=None, width=400, height=None):
#         model_name = model_name or self.IMAGE_MODEL
#         print(f"Generating image for: {user_input}")
#         for attempt in range(5):
#             api_key, _ = get_next_key()
#             print(f"➡️ Using API key: {api_key[:10]} of {_} (Attempt {attempt+1})")
#             llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
#             # Pass **all previous image prompts** as history
#             image_prompts = [h["content"] for h in self.conversation_history if h["type"] == "image"]
#             # Include the current prompt
#             all_prompts = "\n".join(image_prompts + [user_input])
#             messages = [HumanMessage(content=all_prompts)]
#             try:
#                 response = llm.invoke(messages, generation_config={"response_modalities": ["TEXT","IMAGE"]})
#                 image_base64 = self._get_image_base64(response)
#                 if image_base64:
#                     display(Image(data=base64.b64decode(image_base64), width=width, height=height))
#                     # Save only prompt for reference
#                     self._update_history(user_input, image_base64, content_type="image")
#                     return "image_generated"
#                 else:
#                     print("No image returned")
#             except Exception as e:
#                 print(f"Attempt {attempt+1} failed: {e}")
#                 continue
#         raise RuntimeError("All API keys failed or quota exceeded.")

#     # --- Generate text with tools ---
#     # --- Generate text with tools (supports custom system message) ---
#     # ... your existing code ...

#     def generate_text_with_tools(self, user_input, tools, system_override=None, enable_retry=False):
#         """
#         Generate text using tools, using the exact same chat() logic.
#         Tools are passed as a parameter.
#         """
#         self._patch_retry(enable_retry)
#         print(f"\n[CHAT] User input: {user_input}")
#         history = self.conversation_history  # Use actual conversation history
        
#         # Ensure system message is always a valid string
#         system_msg = system_override if system_override else "You are a helpful assistant."
#         messages = [SystemMessage(content=system_msg)]  # <-- corrected line
        
#         # Append previous conversation
#         for h in history:
#             role = HumanMessage if h["role"] == "user" else AIMessage
#             messages.append(role(content=h["content"]))
        
#         # Append current user input
#         messages.append(HumanMessage(content=user_input))
        
#         # Prepare model
#         model_name = self.TEXT_MODEL

#         for attempt in range(11):
#             api_key, _ = get_next_key()
#             print(f"➡️ Using API key: {api_key[:10]} of {_} (Attempt {attempt+1})")

#             try:
#                 llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
                
#                 # Bind tools
#                 model_with_tools = llm.bind_tools(tools, tool_choice="any")
#                 response = model_with_tools.invoke(messages)
                
#                 # Execute tool calls if any
#                 if tool_calls := getattr(response, "tool_calls", None):
#                     messages.append(response)
#                     for tool_call in tool_calls:
                
#                         tool_name = tool_call["name"]
#                         tool_args = tool_call["args"]
            
#                         if callable(tool := next((t for t in tools if t.__name__ == tool_name), None)):
#                             result = tool(**tool_args)
#                             print(f"[TOOL] Called {tool_name} with {tool_args}, returned: {result}")
#                         else:
#                             result = f"No implementation for {tool_name}"
#                             print(f"[TOOL] {result}")
        
#                         messages.append(ToolMessage(
#                             content=str(result),
#                             tool_call_id=tool_call["id"]
#                         ))
#                     # Call model again with tool outputs
#                     response = llm.invoke(messages)
                
#                 # Update conversation history
#                 self.conversation_history.append({"role": "user", "content": user_input})
#                 self.conversation_history.append({"role": "assistant", "content": response.content})
                
#                 print(f"[CHAT] Response is: {response.content}")
#                 return response.content
#             except Exception as e:
#                 print(f"Attempt {attempt+1} failed: {e}")
#                 continue
#         raise RuntimeError("All API keys failed or quota exceeded.")



            
            