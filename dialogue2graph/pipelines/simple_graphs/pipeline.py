from dialogue2graph.pipelines.core.pipeline import Pipeline


pipe = Pipeline(cache_path="./cache", algos=[TopicGenerator, AnotherDialogueSampler])