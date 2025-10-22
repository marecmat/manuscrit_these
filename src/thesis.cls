\NeedsTeXFormat{LaTeX2e}
\newif\ifshowpretty 
\showprettyfalse
\ProvidesClass{src/thesis}[phd thesis template]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\DeclareOption{pretty}{\showprettytrue}
\ProcessOptions\relax
\LoadClass[
    12pt,
    % oneside
    twoside
]{book}

%% Some parameters
\setlength{\headheight}{28pt}
\everymath{\displaystyle}
\RequirePackage{xcolor}     
\definecolor{accent}{HTML}{620059} %{685494}% {a33c34}
\definecolor{accent_title}{HTML}{663361} %{685494}% {a33c34}
\definecolor{titletext}{HTML}{FFFFFF} %{685494}% {a33c34}

\RequirePackage{geometry}   
\geometry{a4paper, left=2.5cm, right=2cm, top=3cm, bottom=3cm}

%% PACKAGES
\RequirePackage[
    natbib, backend=biber, sorting=none, style=numeric-comp, giveninits=true,
    date=year, isbn=false, doi=false,  
    eprint=false, url=false, maxbibnames=6,
    defernumbers=true]{biblatex}
\renewcommand{\cite}{\supercite}

\RequirePackage[labelfont=bf,justification=centering,footnotesize]{caption}
\RequirePackage[explicit]{titlesec}   
\RequirePackage[
    colorlinks=true,
    urlcolor=black,
    citecolor=accent,
    linkcolor=accent,
    ]{hyperref}
\RequirePackage{titletoc,fancyhdr,epigraph,enumitem}%,hyphenat}  
\RequirePackage{amssymb,amsmath,amsthm}  
\RequirePackage{pdfpages,indentfirst,microtype,etoolbox,multicol}  
\RequirePackage{graphicx,tabularx,array,tikz}   
\usetikzlibrary{patterns, patterns.meta, calc, arrows, arrows.meta, external, decorations.pathmorphing, positioning, fit, shapes}
% \tikzexternalize

% \RequirePackage{layout/jabbrv} % https://github.com/compholio/jabbrv
%% FOR CHAPTER GM 
\RequirePackage{cleveref,subcaption}
\RequirePackage{chapitres/article_GM/od}
%% CHAPTER DISP
\newenvironment*{widetext}{}{}

%% epigraph config
\makeatletter
\newlength\epitextskip
\pretocmd{\@epitext}{\em}{}{}
\apptocmd{\@epitext}{\em}{}{}
\patchcmd{\epigraph}{\@epitext{#1}\\}{\@epitext{#1}\\[\epitextskip]}{}{}
\renewcommand*\l@section{\@dottedtocline{1}{1em}{2em}}
\renewcommand*\l@subsection{\@dottedtocline{2}{4em}{3em}}
\renewcommand*\l@subsubsection{\@dottedtocline{3}{9em}{2em}}
\setlength\epigraphrule{0pt}
\setlength\epitextskip{2ex}
\setlength\epigraphwidth{.5\textwidth}

\ifshowpretty
    % Formatage des titres de chapitres, section ... seulement si l'option pretty est passée au package
    \typeout{JOLIE MISE EN PAGE}
        \RequirePackage{./src/faune}
        \newfontfamily\titlestyle{Faune}

        \titleformat{\chapter}[block]
            {\flushleft}
            {\thechapter\tikzexternaldisable
                \begin{tikzpicture}[remember picture, overlay]
                    \fill[accent] (current page.north west) rectangle ([yshift=-6cm]current page.north east);
                    \node[
                        outer sep=0pt,
                        text width=2.5cm,
                        minimum height=2.5cm,
                        font=\color{white}\fontsize{80}{90} \selectfont,
                        align=center
                    ] at ([yshift=-3.2cm, xshift=2.5cm]current page.north west) (num) {\thechapter};
                    \node[
                        outer sep=0pt,
                        inner sep=0pt,
                        anchor=south,
                        font=\color{white}\Large\normalfont
                    ] at ([yshift=3pt]num.north) (chaptertitle){\textls[180]{\textsc{\chaptertitlename}}};
                    \node[
                        text width=14cm, 
                        minimum height=3cm,
                        font=\color{white}\fontsize{24}{24}\titlestyle,
                        anchor=north west,
                    ] at ([yshift=2pt, xshift=1.5cm]chaptertitle.north east){#1};
                \end{tikzpicture}\tikzexternalenable
                \vspace{1.5cm}
            }
            {0pt}
            {}[{\hypersetup{linkcolor=black}
                \vspace*{2pc}{\color{accent}\titlerule\vspace*{1pc}}%
                \startcontents\vbox{\Large\contentsname}\vskip1ex
                \printcontents{l}{1}{\setcounter{tocdepth}{3}}\vspace*{1pc}{\color{accent}\titlerule}
            }]
        \titlespacing*{\chapter}{0pt}{0pt}{40pt}

        \titleformat{name=\chapter,numberless}[display]
        {\flushleft}
        {\thechapter\tikzexternaldisable
        \begin{tikzpicture}[remember picture, overlay]
            \fill[accent] (current page.north west) rectangle ([yshift=-6cm]current page.north east);
            \node[
                font=\color{white}\fontsize{42}{42}\titlestyle,
                anchor=center,
                ] at ([yshift=-3cm]current page.north) {{#1}};
            \end{tikzpicture}\tikzexternalenable
            \vspace{1.5cm}
            }
            {0pt}
            {}
        \titlespacing*{\chapter}{0pt}{0pt}{40pt}

        \titleformat{\section}[block]
            {}
            {}
            {0pt}
            {%
                \vspace{2pt}
                \begin{tabular}[t]{@{}>{\color{accent}\fontsize{18}{18}\titlestyle}p{2.5em}@{\hspace{10pt}}@{\color{accent}\vrule width 2pt\hspace{20pt}}>{\raggedright\arraybackslash}p{0.8\textwidth}@{}}
                    \thesection & {\fontsize{18}{18}\titlestyle\color{black}{#1}}
                \end{tabular}
            }
        \titlespacing*{\section}{0pt}{\baselineskip}{20pt}
        \titleformat{name=\section, numberless}[block]
            {}
            {}
            {0pt}
            {%
                \vspace{2pt}
                \begin{tabular}[t]{@{}>{\color{accent}\fontsize{18}{18}\titlestyle}p{2.5em}@{\hspace{10pt}}@{\color{accent}\vrule width 2pt\hspace{20pt}}>{\raggedright\arraybackslash}p{0.8\textwidth}@{}}
                     & {\fontsize{18}{18}\titlestyle\color{black}{#1}}
                \end{tabular}
            }

        \titleformat{\subsection}
            {\fontsize{16}{16}\color{accent}\titlestyle\bfseries}
            {\thesubsection.}{5pt}{\color{black}{#1}}
        \titlespacing*{\subsection}{0pt}{\baselineskip}{10pt}
        \titleformat{\subsubsection}{\color{accent}\titlestyle\bfseries}{\thesubsubsection.}{5pt}{{#1}}
        \titlespacing*{\subsubsection}{0pt}{\bigskipamount}{5pt}
        \dottedcontents{chapter}[1.5em]{\vspace{0.5\baselineskip}\bfseries}{1.5em}{0pt}
                
        \providecommand{\subtitle}[1]{\apptocmd{\@title}{\par \vspace*{.2em} {\fontsize{18}{18}\titlestyle #1 \par}}{}{}}
        \providecommand{\footer}[1]{\apptocmd{\@title}{\vfill {\fontsize{11}{11}\rmfamily\mdseries #1 \par} \vspace*{-.5em}}{}{}}
        % PAGE DE GARDE
        \renewcommand{\maketitle}{%
            \thispagestyle{empty}
            \tikz[remember picture,overlay] \fill[accent_title, inner sep=0pt] (current page.south east) rectangle (current page.north west);
            \tikz[remember picture,overlay] \node[opacity=.7,inner sep=0pt] at (current page.center){\includegraphics[width=\paperwidth,height=\paperheight]{background.png}};
        \begin{center}
            \vspace*{-2.3cm}
            {\color{titletext}\fontsize{18}{18}\selectfont  
                {Le Mans Université, Le Mans, France}

                {\color{titletext} École Doctorale Sciences de l’Ingénierie et des Systèmes}
                \vskip .2em

                {\color{titletext} Doctoral thesis \, -- \, Speciality: acoustics}

            }%
            \vskip 2.5em%
            
            {\color{titletext}\fontsize{18}{18}\selectfont Presented by}\\[.2em]

            {\color{titletext}\fontsize{22}{22}\titlestyle \@author}\\[.7em]

            {\color{titletext}\fontsize{18}{18}\selectfont submitted for the Doctoral degree}
            \vskip 4cm

            {\color{titletext}\fontsize{26}{26}\titlestyle\@title\par}
        \end{center}%
        \par
        \vskip -1em}

        % \renewcommand*{\cftsubsubsecindent}{2em} % reduce indent of subsubsection titles 
        % Write chapters as chapters in the toc
        \titlecontents{chapter}[0pt]
        {\bfseries}{\textsc{\chaptername}\ \thecontentslabel:\quad}
        {} %numberless format if needed
        {\hfill\contentspage}

        % Fancy page formatting
        \fancyhf{}
        \fancyhead[LE,RO]{\titlestyle}
        \fancyhead[RE]{\titlestyle\nouppercase{\leftmark}}
        \fancyhead[LO]{\titlestyle\nouppercase{\rightmark}}
        \fancyfoot[RE, LO]{\thepage}
        \RequirePackage{emptypage}

        \pagestyle{fancy} 
        \RequirePackage[lining,semibold,scale=1.12]{ebgaramond}
        % \RequirePackage[scale=0.85]{sourcecodepro}
        % \RequirePackage[ebgaramond,vvarbb,subscriptcorrection]{newtxmath}
        \RequirePackage[ebgaramond]{newtxmath}

        % \AtEveryBibitem{\clearfield{title}}
        \renewcommand*{\bibfont}{\fontsize{9}{9}\selectfont}
        \defbibenvironment{bibliography}
        {\enumerate[label={\color{accent}\arabic*.}]
        \setlength{\leftmargin}{-50pt}}
        {\endenumerate}
        {\item}

        \DeclareFieldFormat{pages}{#1}
        \renewbibmacro{in:}{%
        \ifentrytype{article}
            {}
            {\bibstring{in}%
            \printunit{\intitlepunct}}}
        \newbibmacro{string+doiurl}[1]{%
        \iffieldundef{doi}
            {\iffieldundef{url}
            {#1}
            {\href{\thefield{url}}{#1}}}
            {\href{https://doi.org/\thefield{doi}}{#1}}}
        \def\blx@driver#1{%
        \ifcsdef{blx@bbx@#1}
            {\usebibmacro{string+doiurl}{\csuse{blx@bbx@#1}}}
            {\ifcsdef{blx@bbx@*}
            {\blx@warning{%
                No driver for entry type '#1'.\MessageBreak
                Using fallback driver}%
                \usebibmacro{string+doiurl}{\csuse{blx@bbx@*}}}
            {\blx@error
                {No driver found}
                {I can't find a driver for the entry type
                '\abx@field@entrytype'\MessageBreak
                and there is no fallback driver either}}}}
        \setlength{\itemindent}{0pt}
        \renewcommand\labelitemi{{\color{accent}---}}

\else
        \providecommand{\subtitle}[1]{\apptocmd{\@title}{\par \vspace*{.5em} {\fontsize{20}{20}\titlestyle #1 \par}}{}{}}
        \providecommand{\footer}[1]{\apptocmd{\@title}{\vfill {\fontsize{11}{11}\rmfamily\mdseries #1 \par} \vspace*{-.5em}}{}{}}
        \typeout{PAS DE MISE EN PAGE, MODE DRAFT}
        \RequirePackage{newtxmath}

\fi

\usepackage[most]{tcolorbox}
\tcbset{abstract/.style={enhanced, colback=accent!5, colframe=accent, 
  boxrule=0.8pt, left=6pt, right=6pt, top=6pt, bottom=6pt, 
  before skip=1em, after skip=1em, fonttitle=\bfseries, title=Abstract}}

\newenvironment{abstract}
  {\begin{center}\begin{minipage}{0.9\linewidth}
    \begin{tcolorbox}[abstract]\vspace{0.2em}\small}
  {\vspace{0.2em}\end{tcolorbox}\end{minipage}\end{center}}


\defbibenvironment{mespublis}
{\enumerate[label={P\arabic*.}]
    {}
    {\setlength{\labelwidth}{\labelnumberwidth}%
    \setlength{\leftmargin}{\labelwidth}%
    \setlength{\labelsep}{\biblabelsep}%
    \addtolength{\leftmargin}{\labelsep}%
    \setlength{\itemsep}{\bibitemsep}%
    \setlength{\parsep}{\bibparsep}}%
    \renewcommand*{\makelabel}[1]{\hss##1}}
{\endenumerate}
{\item}

\renewcommand*{\thefootnote}{(\arabic{footnote})}
\renewcommand{\figurename}{Fig.}
\renewcommand{\thefigure}{\arabic{chapter}.\arabic{figure}}
\renewcommand\listoffigures{%
  \section*{\listfigurename}%
  \@mkboth{\MakeUppercase\listfigurename}{\MakeUppercase\listfigurename}%
  \@starttoc{lof}%
}
\renewcommand\listoftables{%
  \section*{\listtablename}%
  \@mkboth{\MakeUppercase\listtablename}{\MakeUppercase\listtablename}%
  \@starttoc{lot}%
}
\makeatother