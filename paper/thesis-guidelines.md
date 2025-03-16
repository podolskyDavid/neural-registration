# Structure of the thesis

A typical structure for the chapters of a thesis is this:

## 1. Introduction

You start out with a general text about the topic of your thesis. In particular, here you motivate why your topic is interesting and relevant, what applications it might have, etc. In general, you want to spark interest in the reader to read more. The beginning of this part should be written such that it can also be understood by readers who are not familiar with the area of your topic, and maybe do not know much about mathematics (imagine your friends, flatmates, family, etc.). You can begin with a very broad view and then narrow down more and more until you arrive at the specific topic of your thesis. There, you tell the reader the most relevant previous results on your topic. This is also a preparation for the next subsection in which you describe the results of your thesis.

### 1.1. Contributions of this thesis

You describe on a high level the results of your thesis. You can use this subsection to also give the reader an overview in which chapter one can find which result.

### 1.2. Other related work

If you want, you can use such a subsection in order to mention some important papers that you did not mention earlier, e.g., because they did not fit in the writing flow. You should make sure that you cite all papers that are important and relevant for your topic.

## 2. Preliminaries

In some theses a "preliminaries" section is used to introduce basic concepts like a precise problem definition, notation, or some simple first results. However, do not put too much into such a section. Normally, it is better to introduce concepts in the places where you use them for the first time. Then, the reader does not have to go back to the preliminaries section to recall their definition. If you can write your thesis without such a preliminaries section, this is totally fine and often a good choice.

## 3. Chapter with results

Then you will have several chapters with the actual results of your thesis. How to structure these depends a lot on your actual results, so it is hard to give a general guideline for this.

## n. Conclusion

Often, the last chapter in a thesis is a conclusion section. Here, you give a quick summary of your results. Try not to simply repeat things you wrote before, the reader has already read your thesis up to this point anyway. If you can, look at your results and techniques from a different angle so that you introduce a new perspective. Also, this is a good place to state interesting open problems and directions for future research.

However, depending on your thesis a different structure might be better. So please see this only as one possible option.

# Using LaTeX Effectively

## Use LaTeX right

There are some things to keep in mind when you write your thesis in LaTeX:

- **Do not use "\\"**: In LaTeX, you can start a new line when you type "\\". Use this only if you have a very good reason for that. This happens very rarely :-). The idea behind this is that you tell LaTeX what you mean, and you let LaTeX decide things like where to start a new line or a new page. This is different from, e.g., Microsoft Word where it is common to sometimes start a new line (manually). On the other hand, it is fine that you tell LaTeX where to start a new paragraph since LaTeX cannot know where a new paragraph would make sense in your text. Simply put a blank line to start a new paragraph.

- **Do not use [h] when placing floats**: When you define a float, e.g., a figure, then there is the option "[h]" to force the float to appear "here", i.e., where you define the float in your document. For example, for a figure this would be `\begin{figure}[h]`. Do not use this option, unless you have a very good reason to do this (again, typically you don't :-)). The idea is again that you tell LaTeX that you want to have a float, but you let LaTeX decide where it fits best with the other content. This way, your document looks better since you leave LaTeX the freedom to place the float where it fits best.

- **Use labels**: This is very basic: when you refer to something, e.g., to a lemma or a section, never write the number of the lemma hard-coded in your LaTeX code, but always use labels. For example, instead of writing "Lemma 4" you write "Lemma \ref{lem:name-of-your-lemma}". Otherwise, you need to change the numbering by hand when, e.g., you insert a new lemma before an old lemma. This is a lot of work.

## Use LaTeX efficiently

A thesis in mathematics or computer science is written in LaTeX. As a matter of fact, when you learn LaTeX there is quite a bit of a learning curve. So if you do not have much experience with LaTeX, it is useful if you start learning it soon, before you even start with your thesis. 

There are several programs that make it more comfortable to write LaTeX documents, like Kile, TeXnicCenter, LyX, or Overleaf. My experience is that everybody has her or his favorite programs for writing LaTeX, so you need to try and figure out what works best for you. 

I personally use LyX since there you see directly how your document will look like, including formulae and tables, and you do not need to worry much about LaTeX commands. For drawing figures I use IPE which has all the features you need to draw mathematical figures and it allows you to include LaTeX formulae directly in your figures (just type, e.g., $\alpha$ in a text box). Also, I find that it is very easy to use. 

Some people prefer Overleaf since you do not need to install anything on your local computer. In my experience though, it takes much longer to compile a LaTeX document in Overleaf than on your local computer (which can be annoying for a long document like a thesis). However, installing LaTeX on your local computer is not that difficult, and for your thesis I think it is well worth doing it. 

Under Ubuntu Linux I recommend to simply install the package "tex-live-full" (e.g., type "sudo apt-get install texlive-full" in the console). It includes pretty much any LaTeX package that you will ever need. Under Windows you can install instead for example MiKTeX. However, figure out what works best for you, and make sure that you know LaTeX before you start with your thesis.

Also, you can check whether your university offers a template for theses, following its thesis regulations (like the template of the TU of Munich). Finally, if you have written a large part of text and believe that you will not need it at the end, instead of deleting it it might be better to comment it out or save it in some other file, just in case.

# Writing mathematics

It is important that in your thesis all proofs are mathematically precise and correct. It might take some time for you to polish your proofs until everything is spelled out precisely, every special case is taken care of, and every inaccuracy is fixed. In particular, it might take you a long time compared to the amount of text that you produce. This is normal when writing theses or papers. 

It can easily take you a whole morning to write a proof that is only half a page long at the end. It is important though that you invest this time: only when you write down a formal proof of your claims, you can be absolutely certain that what you believe is really correct. In many cases something seems "intuitively obviously true" but when you write down the formal proof you realize that you missed something. This happens also to very experienced researchers. 

On the other hand, please also give a lot of intuition to the reader. It is hard to read a mathematical text in which the formalism is precise and correct, but in which no intuition is given. For example, it is good to say something about the general structure of a section, a proof, or an algorithm, before you go into details. You may think of your thesis as a story that you tell to somebody. What structure would be good? What should come first and what should come only later?