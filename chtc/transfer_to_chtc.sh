f=MultiTaskRL

cd ../.. # cd just outside the repo
tar --exclude="chtc" \
    --exclude="configs" \
    --exclude='local' \
    --exclude="plotting" \
    --exclude='RESULT' \
    --exclude='results' \
    --exclude='runs' \
    --exclude='.git' \
    --exclude='.idea'  \
    --exclude='results_backup'  \
    --exclude='.venv'  \
    -czvf ${f}.tar.gz $f

scp ${f}.tar.gz whuang369@ap2001.chtc.wisc.edu:/staging/whuang369
rm ${f}.tar.gz