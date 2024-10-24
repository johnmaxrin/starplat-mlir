cd parser
flex avial.l
bison -d avial.y
echo "Done with Parser"

cd ..

g++ -g main.cc parser/lex.yy.c parser/avial.tab.c -o app -I ./ast