import random
from abc import abstractmethod


class Policy:
    @abstractmethod
    def __init__(self, configs):
        pass

    @abstractmethod
    def get_action(self, observation, info):
        pass


class RandomPolicy(Policy):
    def __init__(self, configs):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Pick a product that has quality > 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Random choice a stock idx
                pos_x, pos_y = None, None
                for _ in range(100):
                    # random choice a stock
                    stock_idx = random.randint(0, len(observation["stocks"]) - 1)
                    stock = observation["stocks"][stock_idx]

                    # Random choice a position
                    stock_w, stock_h = stock.shape
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    pos_x = random.randint(0, stock_w - prod_w)
                    pos_y = random.randint(0, stock_h - prod_h)
                    break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}


class GreedyPolicy(Policy):
    def __init__(self, configs):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Pick a product that has quality > 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Loop through all stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = stock.shape
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self.__can_place(stock, (x, y), prod_size):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break

                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def __can_place(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        for x in range(prod_w):
            for y in range(prod_h):
                if stock[pos_x + x][pos_y + y] != 0:
                    return False

        return True
