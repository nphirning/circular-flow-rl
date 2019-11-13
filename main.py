import model
from constants import *

def main():
    m = model.Model(1000)
    m.create_firms(5, rltype=RLType.TRIVIAL)
    m.create_people(10, rltype=RLType.TRIVIAL)
    m.run(1)

    for firm in m.firms:
        print(firm.money, firm.num_goods)
    print("===")
    for person in m.people:
        print(person.money)


if __name__ == "__main__":
    main()