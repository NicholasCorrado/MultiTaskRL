cd ../.. # cd just outside the repo
tar --exclude="chtc" \
    --exclude="configs" \
    --exclude='local' \
    --exclude="plotting" \
    --exclude='results' \
    --exclude='.git' \
    --exclude='.idea'  \
    -czvf MultiTaskRL.tar.gz MultiTaskRL
    
scp MultiTaskRL.tar.gz ncorrado@ap2001.chtc.wisc.edu:/staging/ncorrado
rm MultiTaskRL.tar.gz