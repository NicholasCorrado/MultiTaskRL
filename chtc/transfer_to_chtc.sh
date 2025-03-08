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
    -czvf ${f}.tar.gz $f
    
scp ${f}.tar.gz ncorrado@ap2001.chtc.wisc.edu:/staging/ncorrado
rm ${f}.tar.gz