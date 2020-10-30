LSH : Locality sensitive hashing
--------------------------------------------------------------------------------------------------
***CS F469 IR Assignment - 2***

**Problem Statement**:

We have to implement Local Sensitive Hashing to find out duplicate or similar DNA sequences within the corpus. The steps involved are Shingling, Minhashing and Local Sensitive hashing. The main idea is to hash similar documents into buckets and the documents in a particular bucket have high probability of being similar or duplicates.

**About the project**

Dataset used - [Kaggle-human-data](https://www.kaggle.com/thomasnelson/human-data)

Have a look at the file [Design Architecture](https://github.com/KritiJethlia/LSH/blob/main/Design_Document_Assignment_2.pdf). It includes the concepts used along with the time taken for each implementation step.

Project By:
- **Kriti Jethlia**: Email- <f20180223@hyderabad.bits-pilani.ac.in>
- **Jui Pradhan**: Email- <f20180984@hyderabad.bits-pilani.ac.in>
- **Anusha Agarwal**: Email- <f20180032@hyderabad.bits-pilani.ac.in>
--------------------------------------------------------------------------------------------------
**How to run the code**
--------------------------------------------------------------------------------------------------

1. Clone the repository : https://github.com/KritiJethlia/LSH.git
2. cd LSH
3. Run file: 

              python3 LSH_program.py
  
4. Type your query in the terminal and wait till it returns the similar DNA sequence results :)

---------------------------------------------------------------------------------------------------
**Dependencies/modules used**
---------------------------------------------------------------------------------------------------
- time
- collections
- pandas
- pickle
- Numpy
- random
- operator
- sys
- copy

