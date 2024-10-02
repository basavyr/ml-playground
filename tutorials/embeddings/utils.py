import re


def split_input_by_symbols(input_string: str, symbols: str):
    """
    Takes an input string and splits it into a list of strings,
    applying the split after any of the given symbols (commas or exclamation marks by default),
    and removes any empty strings from the result.
    """
    split_string = re.split(
        f'(?=[{re.escape(symbols)}])|(?<=[{re.escape(symbols)}])', input_string.strip().lower())
    return [x.strip() for x in split_string if x.strip()]


dictionary = [
    # Common words
    "hey", "there", "how", "are", "you", "i", "like", "coffee", "good", "morning",
    "what", "is", "this", "it", "not", "my", "your", "name", "please", "thanks",
    "welcome", "where", "when", "yes", "no", "okay", "today", "tomorrow", "yesterday",
    "time", "want", "need", "help", "know", "happy", "sad", "see", "go", "come",
    "eat", "drink", "love", "book", "read", "write", "car", "bicycle", "phone",
    "computer", "mouse", "keyboard", "screen", "home", "work", "office", "school",
    "friend", "family", "weather", "rain", "sunny", "cold", "hot", "dog", "cat",
    "house", "window", "door", "food", "water", "sleep", "run", "walk", "drive",
    "watch", "movie", "music", "play", "game", "learn", "study", "talk", "listen",
    "clean", "open", "close",

    # Technical words
    "algorithm", "data", "variable", "function", "loop", "condition", "array",
    "list", "dictionary", "tuple", "class", "object", "inheritance", "polymorphism",
    "recursion", "syntax", "compile", "runtime", "error", "exception", "debug",
    "string", "integer", "float", "boolean", "network", "protocol", "server",
    "client", "database", "query", "schema", "table", "index", "join", "key",
    "encryption", "hash", "authentication", "authorization", "firewall", "router",
    "cloud", "API", "endpoint", "request", "response", "cookie", "session",
    "token", "binary", "bit", "byte", "gigabyte", "processor", "memory", "cache",
    "storage", "disk", "virtual", "container", "docker", "kubernetes", "thread",
    "process", "concurrency", "parallelism", "GPU", "CPU", "bandwidth", "latency",

    # ANSI Keyboard Symbols
    "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "=", "+", "[", "]",
    "{", "}", ";", ":", "'", "\"", ",", ".", "/", "\\", "|", "<", ">", "?", "~", "`"
]
