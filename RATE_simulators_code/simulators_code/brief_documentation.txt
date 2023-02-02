Search contexts - the search context stored by the simulated user, the "memory" of the session acc to the simulated user, used as link between search interface and sim.user, stores last queries, documents, snippets viewed...

Search interface - it is the search interface the simulated user interacts with, the interface also uses an underlying search engine to getting results and retreiving documents.

serp impressions - simulator side - making judgements on the serp - determines if the serp is attractive or not (good or bad), also computes path quality using QRELs(serp quality)

sim_config_generator - converts XML input into Dictionary - INPUT

query_generator - Generates query using topic title and description, a query ranked list is made based on likelihood of generating the query given topic.  Also has update model - updated the likelihood model based on search context. 

logger - loggers to log an event, each action can be logged. aso used to check if queries are exhausted, to get progress of the simulation.

Stopping_Decision_maker - makes a decision and returns an Action from the logger action, connected to loggers.
					decision - whether to query or see next snippet

text classifier - simulator side - classifies document as relevant or non relevant, uses TREC Qrels or document likelihood. can 				also update model based on search context. can be used to classify a document or snippet or relevant or irrelevant.

Component generator - Converts configuration dictionary to a Pythons objects of the repective user classes. can also have more 					components

Config reader - vaidates a configuration file using DTD schema, turns the XML file to a Python object dictionary -  same as sim_config_generator.  Uilizes Component generator in the process.









