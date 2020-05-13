set PATH=c:\python37;c:\python37\scripts;%PATH%
pyinstaller --clean -y --hidden-import="tensorflow_core" --hidden-import="sklearn.utils._cython_blas" --hidden-import="sklearn.neighbors.typedefs" --hidden-import="sklearn.neighbors.quad_tree" --hidden-import="sklearn.tree._utils" src\AcousticEmanations.py 
mkdir .\dist\AcousticEmanations\tensorflow\lite\experimental\microfrontend\python\ops
copy deps-exe\ops\* .\dist\AcousticEmanations\tensorflow\lite\experimental\microfrontend\python\ops
copy deps-exe\dll\vcomp140.dll .\dist\AcousticEmanations\
pause

