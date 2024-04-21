#!/usr/bin/bash

mkdir -p code_thesis_DM

cp -r requirements.txt code_thesis_DM
cp -r README.md code_thesis_DM
cp -r src code_thesis_DM
cp -r experiments code_thesis_DM
cp -r papers code_thesis_DM

zip -r code_thesis_DM.zip code_thesis_DM
