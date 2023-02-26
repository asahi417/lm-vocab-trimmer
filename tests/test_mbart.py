import logging
from vocabtrimmer import MT5VocabTrimmer

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

trimmer = MT5VocabTrimmer('lmqg/mbart-large-cc25-jaquad-qg')
trimmer.show_parameter()
out = trimmer.text2text_generation("フェルメールの作品は、疑問作も含め<hl>30数点<hl>しか現存しない。")
print(out)

trimmer.trim_vocab(language='ja', path_to_save='model_test/mbart-large-cc25-jaquad-qg-trim')
trimmer.show_parameter()
out = trimmer.text2text_generation("フェルメールの作品は、疑問作も含め<hl>30数点<hl>しか現存しない。")
print(out)
