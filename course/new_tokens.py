toks = {
    "section"           : ["<SECTION>", "</SECTION>"],
    "content"           : ["<CONTENT>", "</CONTENT>"],
    "course_content"    : ["<CCONTENT>", "</CCONTENT>"],
    "goals"             : ["<GOALS>", "</GOALS>"],
    "general_skills"    : ["<SKILLS>", "</SKILLS>"],
    "education"         : ["<EDUCATION>", "</EDUCTION>"],
    "examination"       : ["<EXAMINATION>", "</EXAMINATION>"],
    "evaluation"        : ["<EVALUATION>", "</EVALUATION>"],
    "prerequisites"     : ["<PREREQ>", "</PREREQ>"],
    "area_of_study"     : ["<AOS>", "</AOS>"],
    "axam_limitations"  : ["<EXAMLIMITS>", "</EXAMLIMITS>"],
    "other"             : ["<OTHER>", "</OTHER>"],
    "newline"           : ["<NL>"],
}
list_of_toks = []
for key in toks:
    for itm in toks[key]:
        list_of_toks.append(itm)
nr_new_toks = len(list_of_toks)

def ret_new_tok_count():
    return nr_new_toks

def ret_new_toks():
    return list_of_toks