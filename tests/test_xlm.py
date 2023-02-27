import logging
from vocabtrimmer import XLMRobertaVocabTrimmer

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# trimmer = XLMRobertaVocabTrimmer('cardiffnlp/xlm-roberta-base-sentiment-multilingual')
# trimmer.show_parameter()
# out = trimmer.text_classification("i'm not even catholic, but pope francis is my dude. like i just need him to hug me and tell me everything is okay. ")
# print(out)
#
# trimmer.trim_vocab(language='en', path_to_save='model/xlm-trimmed')
# trimmer.show_parameter()
# out = trimmer.text_classification("i'm not even catholic, but pope francis is my dude. like i just need him to hug me and tell me everything is okay. ")
# print(out)

trimmer = XLMRobertaVocabTrimmer('bert-base-multilingual-cased')
trimmer.show_parameter()

trimmer.trim_vocab(language='en')
trimmer.show_parameter()
