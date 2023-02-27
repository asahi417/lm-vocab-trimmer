HF_ORG='asahi417'
# mt5-small
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  vocabtrimmer-trimming -m "google/mt5-small" -l "${LA}" --repo-id "${HF_ORG}/mt5-small-trimmed-${LA}"
done

# mbart
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  vocabtrimmer-trimming -m "facebook/mbart-large-cc25" -l "${LA}" --repo-id "${HF_ORG}/mbart-large-cc25-trimmed-${LA}"
done


# mt5: qg models
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  vocabtrimmer-trimming -m "lmqg/mt5-small-${LA}quad-qg" -l "${LA}" --repo-id "lmqg/mt5-small-${LA}quad-qg-trimmed"
  vocabtrimmer-trimming -m "lmqg/mt5-base-${LA}quad-qg" -l "${LA}" --repo-id "lmqg/mt5-small-${LA}quad-qg-trimmed"
done
