Malicious code in DOCX format

In the first task, you should analyze documents in DOCX format and identify malicious code. With this format, each document consists of a series of XML files combined in a zip archive. So the analysis is not difficult. First of all, think about how an attacker can even use malicious code in such a document.

The format for the predictions is as follows:

     data/docx-2017-04/diufgzadsgf.x;1
     data/docx-2017-04/dsiusdfsdaf.x;0
     data/docx-2017-04/fzsdhfksafs.x;0
     data/docx-2017-04/zewrjbakacs.x;0
     ...
The first field is the document's filename and the second field is your prediction. In the example, the document diufgzadsgf.x is classified as malicious. 