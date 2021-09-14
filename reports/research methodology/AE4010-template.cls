\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{AE4010-template}[2020/01/30 v1.1 TU Delft AE4010 class]

\newif\if@dutch
\newif\if@english
\newif\if@nativefonts
\newif\if@print
\newif\if@roman
\newif\if@Hyper
\newif\if@NoHyper
\@romantrue
\DeclareOption{dutch}{\@dutchtrue\@englishfalse}
\DeclareOption{roman}{\@romantrue}
\DeclareOption{noroman}{\@romanfalse}
\DeclareOption{english}{\@dutchfalse\@englishtrue}
\DeclareOption{nativefonts}{\@nativefontstrue}
\DeclareOption{print}{\@printtrue}
\DeclareOption{Hyper}{\@Hypertrue}
\DeclareOption{NoHyper}{\@NoHypertrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{english}
\ExecuteOptions{NoHyper}
\ProcessOptions\relax

\LoadClass[12pt]{article}

\RequirePackage{amsmath}
\RequirePackage{amssymb}

\RequirePackage[nooneline,footnotesize]{caption}
\RequirePackage{fancyhdr}
\RequirePackage[flushmargin,hang]{footmisc}
\RequirePackage{ifpdf}
\RequirePackage{ifxetex}
\ifxetex
    \RequirePackage[xetex]{geometry}
    \RequirePackage[xetex]{graphicx}
    \RequirePackage[xetex, bookmarks=false,
  pdfpagelabels=false,
  hyperfootnotes=false,
  hyperindex=false,
  pageanchor=true]{hyperref}
    \RequirePackage{fontspec}
    \RequirePackage{xltxtra}
    \defaultfontfeatures{Ligatures=TeX}
\else
    \ifpdf
        \RequirePackage[pdftex]{geometry}
        \RequirePackage[pdftex]{graphicx}
        \RequirePackage[pdftex]{hyperref}
    \else
        \RequirePackage[dvips]{geometry}
        \RequirePackage[dvips]{graphicx}
        \RequirePackage[hypertex]{hyperref}
    \fi
\fi

\RequirePackage{metalogo}
\RequirePackage{tikz}
\RequirePackage[noindentafter]{titlesec}
\RequirePackage{titletoc}
\RequirePackage[nottoc]{tocbibind}
\RequirePackage{xcolor}
\RequirePackage{xkeyval}
\RequirePackage[export]{adjustbox}

\ifxetex
    %% If the document is compiled with XeLaTeX, we load the TU Delft house
    %% style fonts: Bookman Old Style (serif) for titles and Arial (sans-serif)
    %% for text. Even if the nativefonts option was specified, we need these
    %% fonts for the cover.
    \newfontfamily\tudtitlefamily[
        Path = fonts/,
        Extension = .ttf%,
    ]{TUDelft-UltraLight}
    \newfontfamily\tudrmfamily[
        Path = fonts/,
        Extension = .ttf,
        BoldFont = *_Bold,
        ItalicFont = *_Italic,
        BoldItalicFont = *_Bold_Italic]{Georgia}
    \newfontfamily\tudsffamily[
        Path = fonts/,
        Extension = .ttf,
        BoldFont = *_Bold,
        ItalicFont = *_Italic,
        BoldItalicFont = *_Bold_Italic]{Arial}
\else
    \@nativefontstrue
    \newcommand*\tudrmfamily{\rmfamily}
    \newcommand*\tudsffamily{\sffamily}
\fi
\if@nativefonts
    %% The native font is Utopia Regular with Fourier math. We use Latin Modern
    %% as a fallback, since Utopia lacks support for sans-serif and monospace.
    \RequirePackage[T1]{fontenc}
    \RequirePackage[utf8]{inputenc}
    \if@roman
      \RequirePackage[thin]{roboto}
      \RequirePackage{fourier}
    \else
      \RequirePackage[thin]{roboto}
      \RequirePackage{helvet}
      \renewcommand{\familydefault}{\sfdefault}
    \fi
    \def\largetitlefont{\fontfamily{Roboto-LF}}
    \def\largetitleseries{\fontseries{t}}
    \def\largetitleshape{\fontshape{n}}
    \def\largetitlestyle{\largetitlefont\largetitleseries\largetitleshape}
    \def\titleshape{\fontshape{n}}
    \if@roman
        \renewcommand{\familydefault}{\rmdefault}
        \def\titlefont{\fontfamily{fla}}
        \def\sectitleshape{\fontshape{n}}
        \def\sectitlestyle{\rmfamily\titleshape\bfseries}%{\familydefault\bfseries}
        \def\chaptitlestyle{\rmfamily\titleshape}
    \else
        \def\titlefont{\fontfamily{fla}}
        \def\sectitlestyle{\sffamily\titleshape\bfseries}%        \def\sectitlestyle{\titleshape\bfseries}
        \def\chaptitlestyle{\sffamily\titleshape}
    \fi
    \def\titlestyle{\titlefont\titleshape}
    \def\headerstyle{\titlefont}
\else
    %% Apart from Bookman Old Style and Arial, we use Courier New for monospace
    %% and Cambria for math.
    \RequirePackage{unicode-math}
    \setmainfont[
        Path = fonts/,
        Extension = .ttf,
        BoldFont = *-Bold,
        ItalicFont = *-Italic,
        BoldItalicFont = *-BoldItalic
    ]{BookmanOldStyle}
    \setsansfont[
        Path = fonts/,
        Extension = .ttf,
        BoldFont = *_Bold,
        ItalicFont = *_Italic,
        BoldItalicFont = *_Bold_Italic]{Arial}
    \setmonofont[
        Path = fonts/,
        Extension = .ttf,
        UprightFont = *MT,
        BoldFont = *-BoldMT,
        ItalicFont = *-ItalicMT,
        BoldItalicFont = *-BoldItalicMT
    ]{CourierNewPS}
    \setmathfont[
        Path = fonts/,
        Extension = .ttf
    ]{CambriaMath}
    %% The default style for text is Tahoma (sans-serif).
    \renewcommand*\familydefault{\sfdefault}
    %% The default style for titles is Bookman Old Style (serif). 
    \def\titlefont{\tudsffamily}
    \def\largetitlefont{\tudtitlefamily}
    \def\titleshape{}
    \def\largetitleshape{\titleshape}
    \def\largetitleseries{}
    \def\largetitlestyle{\largetitlefont\largetitleseries\largetitleshape}
    \def\chaptitlefamily{\tudsffamily}
        \def\titlestyle{\titlefont\titleshape}
%        \def\largetitlestyle{\largetitlefont\titleshape}
        \def\sectitlestyle{\tudsffamily\titleshape\bfseries}
        \def\chaptitlestyle{\chaptitlefamily}
%    \fi
    \def\headerstyle{\titlestyle}
\fi

\geometry{a4paper,hscale=0.75,vscale=0.8}         % Original geometry values of tudelft-report documentclass

\newcommand*{\yourstudynumber}[1]{\gdef\@yourstudynumber{#1}%
}
\newcommand*{\yourmscprofile}[1]{\gdef\@yourmscprofile{#1}%
}
\newcommand*{\yourmsctrack}[1]{\gdef\@yourmsctrack{#1}%
}



\renewcommand{\maketitle}{
    \setkeys{title}{}
    \begin{center}
    {\titlefont\fontsize{28}{34}\selectfont\@title} \\
    \medskip
    {\titlefont\fontsize{11}{13}\selectfont \color{subtitle} By \@author, \@yourstudynumber} \\
    \medskip
    {\titlefont\fontsize{11}{13}\selectfont \color{subtitle} \@yourmscprofile, \@yourmsctrack}
    \end{center}
}


\fancypagestyle{normal}{%
    \fancyhf{}
    \renewcommand*\headrulewidth{0pt}
    \renewcommand*\footrulewidth{0pt}
    \fancyfoot[R]{\titlefont\thepage}
}

\pagestyle{normal}

\definecolor{Word-section}{rgb}{0.27,0.45,0.77}
\definecolor{Word-subsection}{rgb}{0.18,0.33,0.59}
\definecolor{Word-gray}{rgb}{0.35,0.35,0.35}


\if@print
    \colorlet{title}{black}
    \colorlet{section}{black}
    \colorlet{subsection}{black}
    \colorlet{subtitle}{black}
\else
    \colorlet{title}{black}
    \colorlet{section}{Word-section}
    \colorlet{subsection}{Word-subsection}
    \colorlet{subtitle}{Word-gray}
\fi

\titleformat{\section}
    {\fontsize{16}{19}\selectfont\color{section}\sectitlestyle}
    {\bfseries\thesection.\ }
    {0pt}
    {\color{section}}
\titlespacing{\section}{0pt}{\baselineskip}{0pt}

\titleformat{\subsection}
    {\fontsize{13}{16}\selectfont\color{subsection}\sectitlestyle}
    {\bfseries\thesubsection.\ }
    {0pt}
    {\color{subsection}}
\titlespacing{\subsection}{0pt}{\baselineskip}{0pt}


\makeatletter
\let\saved@hyper@linkurl\hyper@linkurl
%\let\saved@hyper@linkfile\hyper@linkfile
\let\saved@hyper@link@\hyper@link@
\AtBeginDocument{%
  % Since the whole document is affected, only the \begin part of
  % environment `NoHyper' is needed.
  \if@Hyper
    ~
  \else
    \NoHyper
  \fi
  \let\hyper@linkurl\saved@hyper@linkurl % needed by \url
  %\let\hyper@linkfile\saved@hyper@linkfile % needed by \href{<file>}
  \let\hyper@link@\saved@hyper@link@ % needed by \href{<url>}
}
\makeatother