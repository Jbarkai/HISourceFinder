
CLUSTER=0
for f in data/training/Input/*.fits; do
    CLS="${f}"
    LBL="data/training/Target/mask_$(echo $f | cut -d'_' -f 2)"
    echo "Cube: ${CLS}"
    echo "Ground-truth: ${LBL}"

    MTO_BASE="../mto-lvq/res/c${CLUSTER}_mto"
    MTO_FITS="${MTO_BASE}.fits"
    MTO_PLOT="${MTO_BASE}.png"
    MTO_LOGS="${MTO_BASE}.log"

    MTO_SCORE="${MTO_BASE}_score.csv"
    MTO_SCORE_PLOT="${MTO_BASE}_score.png"

    if [[ ! -f $MTO_FITS ]]; then
        echo -n "Classifying with MTO ... "
        ../mto-lvq/mto-lvq 1 16 "$CLS" 1 "" "" 32 1 "$CLUSTER" 1 0.0016 0 > "$MTO_LOGS"
        mv segm2D.fits "$MTO_FITS"
        echo "done!"
    fi

    if [[ ! -f $MTO_PLOT ]]; then
        echo -n "Plotting ... "
        python bin/fits.py file "$MTO_FITS" plot -o "$MTO_PLOT" > /dev/null
        echo "done!"
    fi

    if [[ ! -f "$MTO_SCORE" ]]; then
        echo -n "Evaluating... "
        python bin/fits.py match -c "$MTO_SCORE" -o "$MTO_SCORE_PLOT" -t "$CLS" "$LBL" "$MTO_FITS" > /dev/null
        echo "done!"
    fi

    LVQ="cluster${CLUSTER}_mto.joblib"

    echo "Classifier: ${LVQ}"

    LVQ_BASE="../res/c${CLUSTER}_mto_lvq"
    LVQ_FITS="${LVQ_BASE}.fits"
    LVQ_PLOT="${LVQ_BASE}.png"
    LVQ_LOGS="${LVQ_BASE}.log"
    LVQ_LOGS_2="${LVQ_BASE}_training.log"

    LVQ_SCORE="${LVQ_BASE}_score.csv"
    LVQ_SCORE_PLOT="${LVQ_BASE}_score.png"

    if [[ ! -f "$LVQ" ]]; then
        echo -n "Training LVQ classifier..."
        ./mto-lvq 1 16 "$CLS" 1 "$LVQ" "$MTO_FITS" 32 "$CLASSIFIER$CLUSTER" 1 0.0016 0 > "$LVQ_LOGS_2"
        echo "done!"
    fi

    if [[ ! -f "$LVQ_FITS" ]]; then
        echo -n "Classifying with MTO+LVQ ... "
        ./mto-lvq 1 16 "$CLS" 1 "$LVQ" "" 32 1 "$CLASSIFIER$CLUSTER" 1 0.0016 0 > "$LVQ_LOGS"
        mv segm3D.fits "$LVQ_FITS"
        echo "done!"
    fi

    if [[ ! -f "$LVQ_PLOT" ]]; then
        echo -n "Plotting ... "
        python bin/fits.py file "$LVQ_FITS" plot -o "$LVQ_PLOT" > /dev/null
        echo "done!"
    fi

    if [[ ! -f "$LVQ_SCORE" ]]; then
        echo -n "Evaluating ... "
        python bin/fits.py match -c "$LVQ_SCORE" -o "$LVQ_SCORE_PLOT" -t "$CLS" "$LBL" "$LVQ_FITS" > /dev/null
        echo "done!"
    fi
    CLUSTER=$(( CLUSTER + 1 ))
    echo index $CLUSTER
done
