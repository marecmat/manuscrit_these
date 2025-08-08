\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{layout/thesis}[phd thesis template]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass[12pt,twoside]{book}

\RequirePackage{geometry}   
\RequirePackage{xcolor}     
\RequirePackage[
    natbib,
    backend=biber, 
    sorting=none, 
    style=numeric-comp, 
    giveninits=true,
    date=year,
    isbn=false,       
    doi=false,  
    eprint=false,
    url=false,
    maxbibnames=2,
    defernumbers=true,
    % refsection=chapter
]{biblatex}
\RequirePackage[labelfont=bf,justification=centering,footnotesize]{caption}
\RequirePackage[explicit]{titlesec}   
\RequirePackage[
    colorlinks=true,
    urlcolor=black,
    citecolor=accent,
    linkcolor=accent,
    ]{hyperref}
\RequirePackage{titletoc}   
\RequirePackage{fancyhdr}  
\RequirePackage{amssymb}   
\RequirePackage{array} 
\RequirePackage{tabularx}   
\RequirePackage{graphicx}  
\RequirePackage{pdfpages}
\RequirePackage{enumitem}

\RequirePackage{tikz}       
\usetikzlibrary{patterns, patterns.meta, calc, arrows.meta, 
                external, decorations.pathmorphing, positioning,
                fit, shapes, }
\RequirePackage{microtype}  
\setlength{\headheight}{15pt}
\RequirePackage[no-math]{fontspec}

\geometry{a4paper, left=2.5cm, right=2cm, top=2cm, bottom=2cm}
\definecolor{accent}{HTML}{a33c34}

\RequirePackage{./layout/faune}
\newfontfamily\titlestyle{Faune}


\RequirePackage{etoolbox}
\RequirePackage{epigraph}
\makeatletter
\newlength\epitextskip
\pretocmd{\@epitext}{\em}{}{}
\apptocmd{\@epitext}{\em}{}{}
\patchcmd{\epigraph}{\@epitext{#1}\\}{\@epitext{#1}\\[\epitextskip]}{}{}
\makeatother
\setlength\epigraphrule{0pt}
\setlength\epitextskip{2ex}
\setlength\epigraphwidth{.35\textwidth}


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
                text width=15cm, 
                minimum height=3cm,
                font=\color{white}\fontsize{24}{24}\titlestyle,
                anchor=north west,
            ] at ([yshift=2pt, xshift=1.5cm]chaptertitle.north east){#1};
        \end{tikzpicture}\tikzexternalenable
        \vspace{3cm}
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
        ] at ([yshift=-3cm]current page.north) {#1};
    \end{tikzpicture}\tikzexternalenable
    \vspace{3cm}
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
            \thesection & {\fontsize{18}{18}\titlestyle\color{black}#1}
        \end{tabular}
    }
\titlespacing*{\section}{0pt}{\baselineskip}{20pt}
\titleformat{\subsection}{\fontsize{16}{16}\color{accent}\titlestyle\bfseries}{\thesubsection.}{5pt}{\color{black}#1}
\titlespacing*{\subsection}{0pt}{\baselineskip}{10pt}
\titleformat{\subsubsection}{\color{accent}\titlestyle\bfseries}{\thesubsubsection.}{5pt}{#1}
\titlespacing*{\subsubsection}{0pt}{\bigskipamount}{5pt}
\dottedcontents{chapter}[1.5em]{\vspace{0.5\baselineskip}\bfseries}{1.5em}{0pt}

\makeatletter
\renewcommand{\maketitle}{%
    \thispagestyle{empty}
  \begin{flushleft}
    {\fontsize{18}{18}\selectfont  
        {Le Mans Université, Le Mans, France}

        {École Doctorale Sciences de l’Ingénierie et des Systèmes}
        \vskip .5em

        {Doctoral thesis \, -- \, Speciality: acoustics}

    }%
    \vskip 2em%
    
    {\fontsize{18}{18}\selectfont Presented by: }

    {\fontsize{26}{26}\titlestyle \@author}\\[.2em]

    {\fontsize{18}{18}\selectfont submitted for the Doctoral degree}
    \vskip 2.5cm
    \end{flushleft}
        
    \begin{center}
        \fontsize{32}{32}\titlestyle\@title
    \end{center}
    
    \begin{center}%
    \vfill
    Defense scheduled on the ??\\[2em]
    Members of the defense committed:\\[1em]
    {\centering \fontsize{11}{11}\selectfont
    \begin{tabular}{lll}
        Name, Surname          & name, position                                  &{\it Reviewer}\\       
        Name, Surname          & name, position                                  &{\it Reviewer}\\[1em]       
        Name, Surname          & name, position                                  &{\it Examiner}\\       
        Name, Surname          & name, position                                  &{\it Examiner}\\       
        Name, Surname          & name, position                                  &{\it Examiner}\\[1em]       
        J.-P. Groby            & Directeur de Recherche CNRS, LAUM UMR CNRS 6613 &{\it Main supervisor}\\
        O. Dazel               & Professeur des universités, Le Mans Université  &{\it Supervisor}     \\
        V. Romero-Garcìa   & Professeur, Universitat Politécnica de Valencia &{\it Co-supervisor}  \\
    \end{tabular}}
    %{\large \@date}%
  \end{center}%
  \par
  \vskip 4em}
\makeatother


\fancyhf{}
\fancyhead[LE,RO]{\titlestyle}
\fancyhead[RE]{\titlestyle\nouppercase{\leftmark}}
\fancyhead[LO]{\titlestyle\nouppercase{\rightmark}}
\fancyfoot[RE, LO]{\thepage}
\RequirePackage{emptypage}

\pagestyle{fancy} 
\RequirePackage[lining,semibold,scaled=1.05]{ebgaramond}
\RequirePackage[scale=0.85]{sourcecodepro}
\RequirePackage{amsthm}
\RequirePackage[ebgaramond,vvarbb,subscriptcorrection]{newtxmath}

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

% \AtEveryBibitem{\clearfield{title}}
\defbibenvironment{bibliography}
  {\enumerate[label={\arabic*.}]
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
\makeatletter
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
\makeatother

% \renewbibmacro*{issue+date}{%
%   \printtext[]{%
%     \printfield{issue}%
%     \setunit*{\addspace}%
%     \usebibmacro{date}}%
%   \newunit}

\tikzexternalize[optimize=true, optimize command away=\includepdf, prefix=tikzbuild/]

%% MATH COMMANDS 
\renewcommand{\transp}{\mathsf T}
\newcommand{\mbf}{\boldsymbol}
\newcommand{\etal}{\emph{et al.}~}
\newcommand{\inputikz}[1]{\tikzsetnextfilename{#1}\input{figures/#1.tex}}
\newcommand{\numvec}[1]{\underline{#1}}
\newcommand{\nummat}[1]{\underline{\underline{#1}}}
\newcommand{\im}{\mathrm i}
\renewcommand{\exp}[1]{e^{#1}}
\newcommand{\modif}[1]{\textcolor{red!80!black}{#1}}