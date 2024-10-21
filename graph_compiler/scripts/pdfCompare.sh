./bin/AIR --top $1 --src machsuite/$1/$1_no_reuse.cpp --make_pdf --proxy_programl
python scripts/runProgramlPdf.py $1_no_reuse.cpp machsuite/$1/
