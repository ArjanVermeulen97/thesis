Version release: 21-07-05
Check for updates!

Errata from version release 21-07-05
------------------
Changed the hyperlink package options in the CLS to set pageanchor to true. Solves a warning when compiling using Biblatex, thereby speeding up compilation time.


Errata from version release 20-01-30
------------------
Changed the text under "Using the template" to explicitly state to compile via XeLaTeX. Added a similar message in the main file of the document.





Errata from version release 19-08-29
------------------
Added the document options "Hyper" and "NoHyper" (default). 

By default, the "NoHyper" option is specified in the document class and only URLs are active. Changing this to "Hyper" will activate the other links.



Using the template
------------------

This is a template for the AE4010 Research Methodologies course. It is an adaptation of the equivalent Word template, version 2019/20. The template is based on the Latex Article class, heavily influenced by the TU Delft report template obtainable from https://www.tudelft.nl/en/tu-delft-corporate-design/downloads/. The lay-out has been adapted to fit the Word template for consistency, but an all black option is available by specifying the "print" option to the documentclass.

The template is designed to work with XeLaTeX in order to support TrueType and OpenType fonts fonts. This is the recommended compiler. Limited compatibility with other compilers (e.g. pdflatex) is included. It has been designed in Overleaf and support for different (offline) TeX distributions is not guaranteed!

Several custom commands have been added to facilitate the assignment of title information. All values can be assigned in the main.tex file.

Currently the document uses the natbib bibliography engine. However, biblatex support is also available in the template and comments in the main file explain which sections to un(comment) to switch engines. The default bibliography type is author-year, but this can be easily changed in preamble of the main file. It is recommended to use natbib for the author year type and Biblatex for the numbered type. The default bibliography style is set to plainnat for natbib and APA for biblatex.