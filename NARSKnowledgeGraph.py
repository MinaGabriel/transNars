from enum import Enum
import random
from typing import List

import NARSPython.NALGrammar.Sentences
from unidecode import unidecode

from NARSPython.NALGrammar.Terms import Term
import NARSPython.NALInferenceRules
import NARSPython.NALInferenceRules.Syllogistic

import multiprocessing
from multiprocessing import Process, Manager

class NARSKnowledgeGraph:

    class Node:
        class NodeType(Enum):
            Entity = 1, # KG Entity
            Relation = 2 # KG Relation

        def __init__(self, node_type: NodeType, term: Term):
            self.node_type = node_type
            self.term = term
            self.sentences_where_node_is_subject: List[NARSPython.NALGrammar.Sentences.Sentence] = []
            self.sentences_where_node_is_predicate: List[NARSPython.NALGrammar.Sentences.Sentence] = []


    def __init__(self, dataset_directory: str, silent_mode: bool):
        print("Initializing NARS")
        self.dataset_directory: str = dataset_directory
        self.silent_mode = silent_mode

        self.nodes_dict = dict()
        self.entity_ID_to_name = dict()
        self.relation_ID_to_name = dict()
        self.entity_name_to_ID = dict()
        self.relation_name_to_ID = dict()

        self.LoadEntityIDs()
        self.LoadRelationIDs()

    def LoadEntityIDs(self):
        print("Loading Entity IDs into NARS")
        # load all entity IDs
        with open(self.dataset_directory + '/entity2id.txt', encoding="utf8") as f:
            for line in f.readlines():
                pieces = line.split("	")
                if len(pieces) == 1: continue # skip the first line
                name, ID = self.CleanString(pieces[0].rstrip()), int(pieces[1].rstrip())
                self.entity_ID_to_name[ID] = name
                self.entity_name_to_ID[name] = ID

    def LoadRelationIDs(self):
        print("Loading Relation IDs into NARS")
        # load all relation IDs
        with open(self.dataset_directory + '/relation2id.txt', encoding="utf8") as f:
            for line in f.readlines():
                pieces = line.split("	")
                if len(pieces) == 1: continue # skip the first line
                name, ID = self.CleanString(pieces[0].rstrip()), int(pieces[1].rstrip())
                self.relation_ID_to_name[ID] = self.CleanString(name)
                self.relation_name_to_ID[name] = ID

    @staticmethod
    def CleanString(dirty_string: str) -> str:
        return unidecode(dirty_string
                        .replace(",",";")
                        .replace(r"\u0022","\""))



    def CreateNALJudgmentFromLine(self,pieces: List[str]):
        # turn the numeric IDs into strings
        subjectID, predicateID, relationID = int(pieces[0].rstrip()), int(pieces[1].rstrip()), int(pieces[2].rstrip())
        subject, object, relation = self.entity_ID_to_name[subjectID], self.entity_ID_to_name[predicateID], self.relation_ID_to_name[relationID]

        # create NAL belief
        #NAL_judgment = NARSPython.NALGrammar.Sentences.new_sentence_from_string("<<*," + subject + "," + object + ">-->" + relation + ">.")
        NAL_judgment = self.TripletToJudgment(subjectID, predicateID, relationID )
        #print(NAL_judgment)

        NAL_subject_term = NAL_judgment.statement.get_subject_term()
        NAL_predicate_term = NAL_judgment.statement.get_predicate_term()

        # create nodes for the entities if they don't exist
        if NAL_subject_term not in self.nodes_dict: self.nodes_dict[NAL_subject_term] = self.Node(node_type=self.Node.NodeType.Entity, term=NAL_subject_term)
        if NAL_predicate_term not in self.nodes_dict: self.nodes_dict[NAL_predicate_term] = self.Node(node_type=self.Node.NodeType.Entity, term=NAL_predicate_term)

        self.nodes_dict[NAL_subject_term].sentences_where_node_is_subject.append(NAL_judgment)
        self.nodes_dict[NAL_predicate_term].sentences_where_node_is_predicate.append(NAL_judgment)
        #if subject not in nodes_dict: nodes_dict[subject] = Node(node_type=NodeType.Entity, name=subject)
        #if object not in nodes_dict: nodes_dict[object] = Node(node_type=NodeType.Entity, name=object)
        #if relation not in nodes_dict: nodes_dict[relation] = Node(node_type=NodeType.Relation, name=relation)

    def LoadTrainingSetBatch(self, thread_id: int, num_lines_in_batch: int, total_lines):
        start_idx = int(thread_id*num_lines_in_batch)
        end_idx = int(start_idx + num_lines_in_batch)

        for i in range(start_idx,end_idx):
            line = total_lines[i+1]
            pieces = line.split(" ")
            judgment = self.CreateNALJudgmentFromLine(pieces)

    def LoadTrainingSetMultithreaded(self, NUM_TO_LOAD = -1):
        # load the training set as Narsese sentences, and also create nodes
        print("Loading Training Set into NARS")
        largest_num_of_sentences_with_same_predicate = 0
        total_lines = 1



        with open(self.dataset_directory + '/train2id.txt', encoding="utf8") as f:
            i = 0
            lines = f.readlines()
            if NUM_TO_LOAD == -1:
                first_line = lines[0]
                total_lines = int(first_line)
            else:
                total_lines = NUM_TO_LOAD

            num_threads = self.max_threads
            while (total_lines % num_threads > 0):
                num_threads -= 1

            print("Will load " + str(total_lines) + " triples as Judgments using multithreading with " + str(num_threads) + " threads.")

            num_lines_in_batch = total_lines / num_threads

            # launch all the threads, each thread will load a fraction/batch of the training dataset into NARS
            threads = []
            for i in range(num_threads):
                p = Process(target=self.LoadTrainingSetBatch, args=(i,num_lines_in_batch,lines))
                p.start()
                threads.append(p)

            # make sure the threads finish
            for p in threads: p.join()
            pass

    def LoadTrainingSet(self, NUM_TO_LOAD=-1):
        # load the training set as Narsese sentences, and also create nodes
        print("Loading Training Set into NARS")
        largest_num_of_sentences_with_same_predicate = 0
        total_lines = 1
        with open(self.dataset_directory + '/train2id.txt', encoding="utf8") as f:
            i = 0
            for line in f.readlines():
                pieces = line.split(" ")
                if len(pieces) == 1:
                    if NUM_TO_LOAD == -1:
                        total_lines = int(pieces[0])
                    else:
                        total_lines = NUM_TO_LOAD
                    print("Will load " + str(total_lines) + " triples as Judgments.")
                    continue  # skip the first line
                # turn the numeric IDs into strings
                subjectID, predicateID, relationID = int(pieces[0].rstrip()), int(pieces[1].rstrip()), int(
                    pieces[2].rstrip())
                subject, object, relation = self.entity_ID_to_name[subjectID], self.entity_ID_to_name[predicateID], \
                self.relation_ID_to_name[relationID]

                # create NAL belief
                # NAL_judgment = NARSPython.NALGrammar.Sentences.new_sentence_from_string("<<*," + subject + "," + object + ">-->" + relation + ">.")
                NAL_judgment = self.TripletToJudgment(subjectID, predicateID, relationID)
                # print(NAL_judgment)

                NAL_subject_term = NAL_judgment.statement.get_subject_term()
                NAL_predicate_term = NAL_judgment.statement.get_predicate_term()

                # create nodes for the entities if they don't exist
                if NAL_subject_term not in self.nodes_dict: self.nodes_dict[NAL_subject_term] = self.Node(
                    node_type=self.Node.NodeType.Entity, term=NAL_subject_term)
                if NAL_predicate_term not in self.nodes_dict: self.nodes_dict[NAL_predicate_term] = self.Node(
                    node_type=self.Node.NodeType.Entity, term=NAL_predicate_term)

                self.nodes_dict[NAL_subject_term].sentences_where_node_is_subject.append(NAL_judgment)
                self.nodes_dict[NAL_predicate_term].sentences_where_node_is_predicate.append(NAL_judgment)
                # if subject not in nodes_dict: nodes_dict[subject] = Node(node_type=NodeType.Entity, name=subject)
                # if object not in nodes_dict: nodes_dict[object] = Node(node_type=NodeType.Entity, name=object)
                # if relation not in nodes_dict: nodes_dict[relation] = Node(node_type=NodeType.Relation, name=relation)

                i += 1
                if not self.silent_mode: print(
                    "NARS Training Set Load Status: Loading Triple " + str(i) + "/" + str(total_lines))
                else:
                    if i == total_lines // 4:
                        print(
                            "NARS Training Set Load Status: 25%")
                    elif i == total_lines // 2:
                        print(
                            "NARS Training Set Load Status: 50%")
                    elif i == 3*total_lines // 4:
                        print(
                            "NARS Training Set Load Status: 75%")

                if NUM_TO_LOAD != -1 and i >= NUM_TO_LOAD: break

    def AddResultToKnowledgeBase(self, result: NARSPython.NALGrammar.Sentences.Judgment):
        if not self.IsJudgmentAlreadyKnownFromKnowledgeGraph(result):
            self.nodes_dict[result.statement.get_predicate_term()].sentences_where_node_is_predicate.append(result)
            self.nodes_dict[result.statement.get_subject_term()].sentences_where_node_is_subject.append(result)

    def DeriveAnswers(self, question: NARSPython.NALGrammar.Sentences.Question, negative_ratio=1000):
        if negative_ratio <= 0: print("error")

        results = []

        # first, get a known answer (either from the knowledge graph, or potentially derived)
        question_predicate = question.statement.get_predicate_term()
        if question_predicate not in self.nodes_dict: return None
        # num_of_answers = len(nodes_dict[question_predicate].sentences_where_node_is_predicate)
        # random_answer_idx = random.randrange(0,num_of_answers)
        # random_answer = nodes_dict[question_predicate].sentences_where_node_is_predicate[random_answer_idx]

        MAX_ANSWERS = negative_ratio**(1./3.) + 1
        answers = 0
        for answer in self.nodes_dict[question_predicate].sentences_where_node_is_predicate:
            # from the answer's subject, get another statement with the same subject but a different predicate/relation
            answer_subject = answer.statement.get_subject_term()

            MAX_RELATIONS = negative_ratio**(1./3.) + 1
            relations = 0
            for sentence_with_new_relation in self.nodes_dict[answer_subject].sentences_where_node_is_subject:
                if sentence_with_new_relation.statement == answer.statement: continue
                # now, use the relation predicate to find a sentence with that relation but a *different* subject
                # random_relation_predicate = random_relation.statement.get_predicate_term()
                # num_of_others = len(nodes_dict[random_relation_predicate].sentences_where_node_is_predicate)
                # if num_of_others <= 1: return None
                # random_other_idx = random.randrange(0,num_of_others)
                # random_other = nodes_dict[random_relation_predicate].sentences_where_node_is_predicate[random_other_idx]
                # if random_other.statement == random_relation.statement:
                #     # we picked the same judgment, so pick a different one
                #     random_other_idx -= 1
                #     random_other = nodes_dict[random_relation_predicate].sentences_where_node_is_predicate[random_other_idx]

                MAX_DERIVATIONS = negative_ratio**(1./3.) + 1
                derivations = 0

                for other_subject in self.nodes_dict[sentence_with_new_relation.statement.get_predicate_term()].sentences_where_node_is_predicate:
                    j1 = sentence_with_new_relation
                    j2 = other_subject
                    if j1.statement == j2.statement: continue

                    similarity = NARSPython.NALInferenceRules.Syllogistic.Comparison(j1, j2)

                    result = NARSPython.NALInferenceRules.Syllogistic.Analogy(answer, similarity)
                    self.AddResultToKnowledgeBase(result)
                    results.append(result)
                    derivations += 1
                    if derivations >= MAX_DERIVATIONS: break

                relations += 1
                if relations >= MAX_RELATIONS: break
            answers += 1
            if answers >= MAX_ANSWERS: break
        # compute a similarity relation
        # now we have 2 sentences with different subjects, but the same predicate, so we compute
        # {J1=<P-->M>,J2=<S-->M>}:- <S<->P> (F'_comparison(j1,j2), aka F_comparison(j2,j1))

        return results


    def IsJudgmentAlreadyKnownFromKnowledgeGraph(self,judgment: NARSPython.NALGrammar.Sentences.Judgment):
        for sentence in self.nodes_dict[judgment.statement.get_subject_term()].sentences_where_node_is_subject:
            if judgment.statement == sentence.statement:
                return True
        return False

    def TripletToJudgment(self, subjectID: int, objectID: int, relationID: int):
        subject, object, relation = self.entity_ID_to_name[subjectID], self.entity_ID_to_name[objectID], self.relation_ID_to_name[relationID]
        return NARSPython.NALGrammar.Sentences.new_sentence_from_string("<" + subject + "-->" + "(/," + relation + ",_," + object + ")>.")

    # put -1 to make it a variable question
    def TripletToQuestion(self, subjectID: int = -1, objectID: int = -1, relationID: int = -1):
        subject = self.entity_ID_to_name[subjectID] if subjectID != -1 else "?s"
        object = self.entity_ID_to_name[objectID] if objectID != -1 else "?o"
        relation = self.relation_ID_to_name[relationID] if relationID != -1 else "?r"

        return NARSPython.NALGrammar.Sentences.new_sentence_from_string("<" + subject + "-->" + "(/," + relation + ",_," + object + ")>?")

    def RunTestSet(self):
        # load the test set as Narsese sentences, and also create nodes
        print("Trying Test Set")
        score = 0
        total = 0
        with open(self.dataset_directory + '/test2id.txt', encoding="utf8") as f:
            line_count = 0
            for line in f.readlines():
                pieces = line.split(" ")
                if len(pieces) == 1:
                    total_lines = pieces[0]
                    continue # skip the first line
                # turn the numeric IDs into strings
                subjectID, predicateID, relationID = pieces[0].rstrip(), pieces[1].rstrip(), pieces[2].rstrip()

                ground_truth_answer = self.TripletToJudgment(subjectID, objectID, relationID)
                #print(answer)

                question = NARSPython.NALGrammar.Sentences.new_sentence_from_string("< ?x -->" + "(/," + relation + ",_," + object + ")>?")
                NAL_predicate_term = question.statement.get_predicate_term()

                if not self.silent_mode: print(str(line_count) + "/" + str(total_lines) + ": " +  str(question))


                if NAL_predicate_term not in self.nodes_dict:
                    print("This predicate was not found in the graph")
                else:
                    print("In the Knowledge Graph, there are " + str(len(self.nodes_dict[NAL_predicate_term].sentences_where_node_is_predicate))
                          + " sentences with this predicate.")


                correct_answer_found = False
                results = self.DeriveAnswers(question)
                if results is None:
                    print("No answers...")
                else:
                    i = 0
                    for result in results:
                        if not self.silent_mode: print(str(i) + "/" + str(len(results)) + ": RESULT : " + str(result))

                        if result.statement == ground_truth_answer.statement:
                            correct_answer_found = True

                        i += 1

                    if correct_answer_found: score += 1

                line_count += 1
                total += 1

        print("TOTAL ACCURACY: " + str(score*100/total) + "%")