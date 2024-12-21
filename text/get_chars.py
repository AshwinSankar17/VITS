import string

charset_ranges = {
            'asm': ('\u0980', '\u09FF'),  # Assamese
            'ben': ('\u0980', '\u09FF'),  # Bengali
            'brx': ('\u0900', '\u09FF'),  # Bodo 
            'doi': ('\u0900', '\u09FF'),  # Dogri
            'guj': ('\u0A80', '\u0AFF'),  # Gujarati
            'hin': ('\u0900', '\u09FF'),  # Hindi
            'kan': ('\u0C80', '\u0CFF'),  # Kannada
            'kas': ('\u0C80', '\u0CFF'),  # Kashmiri
            'kok': ('\u0900', '\u09FF'),  # Hindi
            'mai': ('\u0900', '\u09FF'),  # Hindi
            'mal': ('\u0D00', '\u0D7F'),  # Malayalam
            'mar': ('\u0900', '\u09FF'),  # Marathi
            'nep': ('\u0900', '\u097F'),  # Nepali
            'ori': ('\u0B00', '\u0B7F'),  # Odia
            'pan': ('\u0A00', '\u0A7F'),  # Punjabi
            'san': ('\u0900', '\u09FF'),  # Sanskrit
            'snd': ('\u0900', '\u097F'),  # Sindhi
            'sat': ('\u1C00', '\u1C4F'),  # Santali
            'tam': ('\u0B80', '\u0BFF'),  # Tamil
            'tel': ('\u0C00', '\u0C7F'),  # Telugu
            'urd': ('\u0900', '\u097F'),  # Urdu
        }

all_chars = ''
for script, (start, end) in charset_ranges.items():
    all_chars += ''.join(chr(i) for i in range(ord(start), ord(end) + 1))

marathi_chars = ''.join(sorted(set(all_chars)))

# Convert to list of characters
marathi_chars_list = list(marathi_chars)

marathi_chars_list = list(filter(str.isprintable, marathi_chars_list))

# Write Marathi characters to a file
with open('marathi_characters.txt', 'w', encoding='utf-8') as file:
    file.write(str(marathi_chars_list))

print("Marathi characters saved to 'marathi_characters.txt'.")