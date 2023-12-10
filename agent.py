import types
import copy
from openai import OpenAI

import os
import json
import typing
import jsonschema
from .billing import update_billing
from .models import *
from pprint import pprint

# NOTE: https://openai.com/blog/function-calling-and-other-api-updates
# According to this the reply is supposed to be sent back as role "function" and
# the return value is supposed to be in "content" parameter as string encoded
# json.
#
# I don't believe the return value description is used.

client = None
api_key = None

# Call this first to set the API key.
def set_openai_api_key(key):
  global api_key
  api_key = key


# Singleton, uff.
def get_default_openai_client():
  assert api_key != None
  global client
  if client is None:
    client = OpenAI(api_key=api_key)

  return client


def indent_len(s, tab=4):
  cnt = 0
  for ch in s:
    if ch == ' ':
      cnt += 1
    elif ch == '\t':
      cnt += tab
    else:
      return cnt

def simple_arg_type_to_json_schema_name(t):
  if t is str:
    return "string"
  if t is dict:
    return "object"
  if t is int:
    return "number"
  if t is float:
    return "number"
  if t is list:
    return "array"
  if t is bool:
    return "boolean"

  raise Exception(f"Should never get here {t}")

def arg_type_to_json_schema(arg_info, t):
  if t in {str, dict, int, float, list, bool}:
    arg_info["type"] = simple_arg_type_to_json_schema_name(t)
    return

  if type(t) is typing._GenericAlias:
    # A bit more complex container. Only few of these are supported and support
    # for these should be a bit better.
    #
    # Some developer notes (will be useful to improve these later):
    # Apparently this is how to check if something is typing.List[str] etc.
    # >>> a.__annotations__["x"] == List[str]
    # True
    # >>> a.__annotations__["x"] == List[int]
    # False
    # a.__annotations__["x"].__args__[0]
    # <class 'str'>
    # >>> typing.get_args(a.__annotations__["x"])
    # (<class 'str'>,)

    if t._name == "List":
      arg_info["type"] = "array"

      item_type = t.__args__[0]
      if item_type in {str, dict, int, float, list, bool}:
        arg_info["items"] = {
            "types": simple_arg_type_to_json_schema_name(item_type)
        }
        return

    # TODO: Perhaps some object support in the future?

    # Fall back to raising exception.

  raise Exception(f"Can't convert type to jsonschema. Unknown type {t}")

def google_style_docstring_to_func_desc(func_desc, doc, annotations):
  def add_param(arg_name, arg_desc):
    if arg_name is None:
      return

    arg_info = {
      "description": arg_desc
    }

    arg_type_to_json_schema(arg_info, annotations[arg_name])

    func_desc["parameters"]["properties"][arg_name] = arg_info

    # TODO: if not optional
    # I guess this could be detected if we would have arg position and
    # number of non-default args.
    # number_of_non_default_args = f.__code__.co_argcount - len(f.__defaults__)
    func_desc["parameters"]["required"].append(arg_name)

  sections = doc.split("\n\n")
  function_description = sections[0].strip()
  func_desc["description"] = function_description
  for section in sections[1:]:
    lines = section.strip().splitlines()
    section_type = lines[0].strip()
    section_lines = lines[1:]
    match section_type:
      case "Args:":
        arg_indent = indent_len(section_lines[0])
        arg_name = None
        arg_desc = None
        for ln in section_lines:
          if indent_len(ln) == arg_indent:
            add_param(arg_name, arg_desc)

            arg_split = ln.split(":", 1)
            arg_name = arg_split[0].strip()
            arg_desc = arg_split[1].strip()
          else:
            arg_desc += f" {ln.strip()}"
        add_param(arg_name, arg_desc)
      case "Returns:":
        return_desc = '\n'.join([ln.strip() for ln in section_lines])
        func_desc["description"] += f"\nReturns: {return_desc}"


def extract_func_desc(f):
  # Note: Function description is just a JSON Schema.
  func_desc = {
    "name": f.__name__,
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  }

  google_style_docstring_to_func_desc(func_desc, f.__doc__, f.__annotations__)

  return func_desc

def chatgpt_api(method):
  func_desc = extract_func_desc(method)
  setattr(method, "chatgpt_api", func_desc)

  return method

class ChatGPT_Agent:
  def __init__(self, model):
    self.chatgpt_model = model
    self.chatgpt_name = "generic_agent"
    self.chatgpt_api_functions = {}
    self.debug = False
    self.the_end = False  # A function can set this to get out of the loop.

    for field in dir(self):
      f = getattr(self, field)
      if type(f) is not types.MethodType:
        continue
      func_desc = getattr(f, "chatgpt_api", None)
      if func_desc is None:
        continue
      self.chatgpt_api_functions[func_desc["name"]] = (f, func_desc)

    self.chatgpt_messages = []
    self.chatgpt_functions = [func_desc for f, func_desc in
        self.chatgpt_api_functions.values()
    ]

  def load_state_from_json_file(self, fname):
    """Loads the message log/state from the JSON file and replaces the current
    one with the loaded one.

    Returns:
      The whole loaded JSON in case the agent wants to load something more
      manually.

    """
    with open(fname) as f:
      state = json.load(f)

    chatgpt_messages = state.get("chatgpt_messages")

    if chatgpt_messages is None:
      raise Exception("Missing 'chatgpt_messages' key in loaded state.")

    self.chatgpt_messages = chatgpt_messages
    return state

  def store_state_in_json_file(self, fname):
    with open(fname, "w") as f:
      json.dump({
          "chatgpt_messages": self.chatgpt_messages
      }, f, indent=4)

  def set_name(self, name):
    # This name is used in log name, so make it filesystem friendly.
    self.chatgpt_name = name

  def add_system_message(self, msg):
    self.add_message("system", msg)

  def add_message(self, role, content):
    self.chatgpt_messages.append({
        "role": role,
        "content": content.strip()
    })

  def reset(self):
    self.chatgpt_messages = []

  def invoke(self, msg=None, role="user", call_function=None):
    self.the_end = False
    replies = []

    while not self.the_end:  # TODO: limit this?
      # TODO: Try except.

      response = self.invoke_worker(msg, role, call_function)

      if response.message.content:
        replies.append(response.message.content)
        if self.debug:
          print(f"{self.chatgpt_name} \x1b[1;31mSays: \x1b[m\x1b[31m{response.message.content}\x1b[m")

      # Resetting call_function since after the first iteration we don't want
      # ChatGPT to call the function anymore.
      # Same with msg.
      # TODO: This might need to be configurable?
      call_function = None
      msg = None

      finish_reason = response.finish_reason

      if finish_reason == "stop":
        return replies

      if finish_reason == "function_call":
        # Function call already happened at this point and the result has been
        # added to the context, so we just-recall ChatGPT with no new messages.
        continue

      raise Exception(f"No idea what this finish reason is: {finish_reason}")


  def invoke_worker(self, msg=None, role="user", call_function=None):
    # Add new message to the context, if any.
    # E.g. when a function call was invoked in a non-forced way, it might make
    # sense to call ChatGPT again with no message.
    if msg is not None:
      self.chatgpt_messages.append({
          "role": role,
          "content": msg
      })

    # Resolve what function_call should be used.
    function_call = "auto"
    if call_function is not None:
      if call_function not in self.chatgpt_api_functions:
        raise Exception(f"Unknown function {fname}")
      function_call = {
          "name": call_function
      }

    # Pass the context to ChatGPT.
    #pprint(self.chatgpt_functions)

    if not self.chatgpt_functions:
      r = get_default_openai_client().chat.completions.create(
          model=self.chatgpt_model,
          messages=self.chatgpt_messages
      )
    else:
      r = get_default_openai_client().chat.completions.create(
          model=self.chatgpt_model,
          messages=self.chatgpt_messages,
          functions=self.chatgpt_functions,
          function_call=function_call
      )

    # TODO: add logging
    #pprint(r)

    # This code is weird because of porting from openai 0.xx to 1.xx.
    # Could use some refactoring.
    response = r.choices[0]
    response_message = response.message
    response_content = response_message.content

    # Add reply to the context.
    self.chatgpt_messages.append(
        response_message.model_dump(exclude_unset=True)
    )

    cost, total_cost = update_billing(
        self.chatgpt_model, r.usage.prompt_tokens, r.usage.completion_tokens
    )

    # Was this a function call attempt?
    func_call = response_message.function_call
    if func_call:

      # Fun fact! LLM agents might try to call non-existing functions, or may
      # get the arguments wrong. Or JSON structure wrong.
      # TODO: Do I want to retry in that case? Maybe even remove the last
      # message? Should there be a retry counter?

      # TODO: json.decoder.JSONDecodeError
      func_args = json.loads(func_call.arguments, strict=False)
      func_name = func_call.name

      #print(f"Attempting to call function: {func_name}")
      #pprint(func_args)

      func, func_desc = self.chatgpt_api_functions.get(func_name)
      if func_desc is None:
        # TODO: change this to some internal exception
        raise Exception(
            f"Agent tried to call non-existing function {func_name}."
        )

      # This will raise an exception ValidationError.
      jsonschema.validate(instance=func_args, schema=func_desc)

      if self.debug:
        print(f"{self.chatgpt_name} \x1b[1;33mCalling function: {func_name}({func_args})\x1b[m")

      # Call the function.
      # TODO: This sometimes throws:
      # TypeError: <function name> missing 1 required positional argument: 'name'
      ret = func(**func_args)
      self.chatgpt_messages.append({
          "role": "function",
          "name": func_name,
          "content": json.dumps(ret)
      })

    return response
