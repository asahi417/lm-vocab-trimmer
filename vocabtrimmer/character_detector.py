import string
import re
import unicodedata as ud

__all__ = "filter_vocab"

currency = ['؋', 'L', '֏', 'ƒ', '$', '$', 'ƒ', '₼', '$', '৳', '$', '$', '$', '฿', 'P', '$', '$', '¥', '$', '₡', '$',
            '₱', '$', '£', 'Ξ', '€', '$', '£', '£', '₾', '£', '₵', '£', 'D', 'Q', '$', '$', 'L', 'G', '₪', '£', '₹',
            '﷼', '£', '¥', '៛', '₩', '₩', '$', '₸', '₭', '£', '₨', '$', 'M', 'Ł', 'K', '₮', '₨', '$', '$', '₦', '₨',
            '$', '﷼', 'K', '₱', '₨', '﷼', '￥', '₽', '﷼', '$', '₨', '$', '£', 'S', '$', '£', '$', '£', 'E', '฿', 'T',
            '₤', '₺', '$', '₴', '$', '₫', 'Ƀ', '$', '₣', '﷼', 'R']
c_latin = '¿¡'
c_en = string.ascii_lowercase
c_num = '0123456789'
c_sp = string.punctuation + '—▁–▁́°×»' + "".join(currency)
range_ja = [
        {"from": ord(u"\u3300"), "to": ord(u"\u33ff")},  # compatibility ideographs
        {"from": ord(u"\ufe30"), "to": ord(u"\ufe4f")},  # compatibility ideographs
        {"from": ord(u"\uf900"), "to": ord(u"\ufaff")},  # compatibility ideographs
        {"from": ord(u"\U0002F800"), "to": ord(u"\U0002fa1f")},  # compatibility ideographs
        {'from': ord(u'\u3040'), 'to': ord(u'\u309f')},  # Japanese Hiragana
        {"from": ord(u"\u30a0"), "to": ord(u"\u30ff")},  # Japanese Katakana
        {"from": ord(u"\u2e80"), "to": ord(u"\u2eff")},  # cjk radicals supplement
        {"from": ord(u"\u4e00"), "to": ord(u"\u9fff")},
        {"from": ord(u"\u3400"), "to": ord(u"\u4dbf")},
        {"from": ord(u"\U00020000"), "to": ord(u"\U0002a6df")},
        {"from": ord(u"\U0002a700"), "to": ord(u"\U0002b73f")},
        {"from": ord(u"\U0002b740"), "to": ord(u"\U0002b81f")},
        {"from": ord(u"\U0002b820"), "to": ord(u"\U0002ceaf")}  # included as of Unicode 8.0
    ]
range_zh = [
        {"from": ord(u"\u3300"), "to": ord(u"\u33ff")},  # compatibility ideographs
        {"from": ord(u"\ufe30"), "to": ord(u"\ufe4f")},  # compatibility ideographs
        {"from": ord(u"\uf900"), "to": ord(u"\ufaff")},  # compatibility ideographs
        {"from": ord(u"\U0002F800"), "to": ord(u"\U0002fa1f")},  # compatibility ideographs
        {"from": ord(u"\u2e80"), "to": ord(u"\u2eff")},  # cjk radicals supplement
        {"from": ord(u"\u4e00"), "to": ord(u"\u9fff")},
        {"from": ord(u"\u3400"), "to": ord(u"\u4dbf")},
        {"from": ord(u"\U00020000"), "to": ord(u"\U0002a6df")},
        {"from": ord(u"\U0002a700"), "to": ord(u"\U0002b73f")},
        {"from": ord(u"\U0002b740"), "to": ord(u"\U0002b81f")},
        {"from": ord(u"\U0002b820"), "to": ord(u"\U0002ceaf")}  # included as of Unicode 8.0
    ]
hangul_ranges = [
        range(0xAC00, 0xD7A4),  # Hangul Syllables (AC00–D7A3)
        range(0x1100, 0x1200),  # Hangul Jamo (1100–11FF)
        range(0x3130, 0x3190),  # Hangul Compatibility Jamo (3130-318F)
        range(0xA960, 0xA980),  # Hangul Jamo Extended-A (A960-A97F)
        range(0xD7B0, 0xD800),  # Hangul Jamo Extended-B (D7B0-D7FF)
    ]


def norm(target): return ud.normalize('NFKC', target.lower())


def single_alphabet(target: str): return len(re.findall(fr'[{c_en}]', norm(target))) == 1


def cd_all_symbol(target: str): return all([c in c_sp + c_num for c in norm(target)])


def cd_eu(target: str):
    if bool(re.search(fr'[{c_latin}]', norm(target))):
        return True

    latin_letters = {}

    def is_latin(uchr):
        try:
            return latin_letters[uchr]
        except KeyError:
            return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))

    return any(is_latin(c) for c in norm(target) if c.isalpha())  # isalpha suggested by John Machin


def cd_en(target: str): return bool(re.search(rf"[{c_en}]", norm(target)))


def cd_ru(target: str):
    if single_alphabet(target):
        return True
    return bool(re.search('[а-яА-Я]', norm(target)))


def cd_ja(target: str):
    if single_alphabet(target):
        return True
    if bool(re.search('[。、！？]', norm(target))):
        return True
    return any(any([r["from"] <= ord(c) <= r["to"] for r in range_ja]) for c in norm(target))


def cd_zh(target: str):
    if single_alphabet(target):
        return True
    return any(any([r["from"] <= ord(c) <= r["to"] for r in range_zh]) for c in norm(target))


def cd_ko(target: str):
    if single_alphabet(target):
        return True
    return any(any(ord(c) in r for r in hangul_ranges) for c in norm(target))


def filter_vocab(vocab, language):
    if language.lower() == 'en':
        return {k: v for k, v in vocab.items() if cd_en(k) or cd_all_symbol(k)}
    elif language.lower() in ['eu', 'it', 'fr', 'de', 'pt', 'es']:
        return {k: v for k, v in vocab.items() if cd_eu(k) or cd_all_symbol(k)}
    elif language.lower() == 'ko':
        return {k: v for k, v in vocab.items() if cd_ko(k) or cd_all_symbol(k)}
    elif language.lower() == 'ja':
        return {k: v for k, v in vocab.items() if cd_ja(k) or cd_all_symbol(k)}
    elif language.lower() == 'zh':
        return {k: v for k, v in vocab.items() if cd_zh(k) or cd_all_symbol(k)}
    elif language.lower() == 'ru':
        return {k: v for k, v in vocab.items() if cd_ru(k) or cd_all_symbol(k)}
    raise ValueError(f'Invalid language: {language}')
