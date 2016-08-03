
def wrapper_for_nb_in_sklearn(data, current_state_to_predict):
    """
        Import an already-built implementation, train it on the data,
    and return the class predicted given the current state.

        Note that the last column of data is assumed to be the variable
    to predict, and the order
    """
    from sklearn import preprocessing
    from sklearn.naive_bayes import MultinomialNB

#     leList = [preprocessing.LabelEncoder().fit([x[i] for x in data]) for i in range(len(data[0]))]
#     for x in leList:
#         print list(x.classes_)
#     for x in data:
#         print x
#     factors = [[leList[i].transform(x[i]) for i in range(len(x)-1)] for x in data]
#     for x in factors:
#         print x
#     states = [leList[-1].transform(x[-1]) for x in data]
#     for x in states:
#         print x
    
    le = preprocessing.LabelEncoder()
    all_states = [item for sublist in data for item in sublist]
    le.fit(all_states)
    print list(le.classes_)
    labelled_data = [[le.transform(x) for x in row] for row in data]
    factors = [row[:-1] for row in labelled_data]
    for x in factors:
        print x
    states = [[row[-1]] for row in labelled_data]
    for x in states:
        print x

    clf = MultinomialNB()
    clf.fit(factors, states)

    state = current_state_to_predict
    print state
    state_transformed = [le.transform(state[i]) for i in range(len(state))]
    print state_transformed
    prediction = clf.predict(state_transformed)
    print clf.predict_proba(state_transformed)

    return le.inverse_transform(prediction)[0]
