#####################
# FOUNDATION MODELS #
#####################
# xlm-r
for SIZE in "base" "large"
do
  for LA in 'fr' 'de' 'es' 'it' 'ar' 'pt'
  do
    for TARGET in 15000 30000 45000
    do
      MODEL="xlm-roberta-${SIZE}-trimmed-${LA}-${TARGET}"
      vocabtrimmer-trimming -m "xlm-roberta-${SIZE}" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
    done
  done
done