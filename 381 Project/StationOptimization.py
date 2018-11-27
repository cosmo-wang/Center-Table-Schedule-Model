# This class encodes information about one station at Center Table
# and optimize the schedule for that station.

from gurobipy import *
from sets import Set
import copy


class Station:
    def __init__(self, station_name, hours, shifts, speed):
        """
        Create a new food station.
        :param station_name: name of the station
        :param hours: opening hours of the station
        :param shifts: all possible shift of the station
        :param speed: speed of serving customers at this station
        """
        self.station_name = station_name
        self.hours = hours
        self.shifts = shifts
        self.speed = speed

    def get_name(self):
        """
        Get name of the station
        :return: name of the station
        """
        return station_name

    def get_hours(self, day):
        """
        Returns opening hours of the station.
        :param day: specifies which day is being asked opening hours for
        :return: a list of tuple representing opening hours
        """
        return copy.copy(self.hours.get(day))

    def get_shifts(self):
        """
        Returns all shifts of the station.
        :return: all shifts of the station
        """
        return copy.copy(self.shifts)

    def get_shift_num(self):
        """
        Get the number of all possible shifts of the station
        :return: total number of different shifts of the station
        """
        return len(self.shifts)

    def get_speed(self):
        """
        Get serving speed of the station given the number of people working.
        :param people_num: number of people working
        :return: speed of serving customers at this station
        """
        return self.speed


def load_prob(dict_list, filename):
    """
    Load data about probability of a given customer going to a station at any time.
    :param dict_list: list of dictionary representing probability of each station.
                      Each dictionary maps a time to a probability
    :param filename: name of the file storing the data
    """
    prob = open("data/" + filename)
    i = 0
    for line in prob:
        if not line.startswith("#"):
            values = line.split(",")
            for j in range(len(dict_list)):
                dict_list[j][7 + 0.25 * i] = float(values[j]) / 100
            i += 1
    prob.close()


def load_customer_count(filename):
    """
    Load data about number of customers coming in at given time, time separated by 15 mins.

    :param filename: name of the file storing the data
    :return: a dictionary maps from time of a day to number of customers coming in at that time
    """
    customer_count = {}
    data = open("data/" + filename)
    i = 0
    for line in data:
        if not line.startswith("#"):
            customer_count[7 + 0.25 * i] = int(line)
            i += 1
    data.close()
    return customer_count


def load_hours(filename):
    """
    Load data about opening hours of a station.
    :param filename: name of the file storing the data
    :return: a dictionary with data about opening hours. Map from day of a week to a list of opening hours
                 throughout the day. Each period of opening hours is a tuple in the form: (open_time, close_time)
    """
    hours = {}
    data = open("data/" + filename)
    for line in data:
        if not line.startswith("#"):
            day = line.split(":")[0]
            day_hours = line.split(":")[1].split()
            parsed_day_hours = []
            for hour in day_hours:
                parsed_day_hours.append((float(hour.split(",")[0]), float(hour.split(",")[1].rstrip())))
            hours[day] = parsed_day_hours
    data.close()
    return hours


def load_shifts(filename):
    """
    Load data about all possible shifts of a station.
    :param filename: name of the file storing the data
    :return: a list of shifts. Shifts are represented by tuples in the form: (start_time, end_time)
    """
    shifts = []
    data = open("data/" + filename)
    for line in data:
        if not line.startswith("#"):
            shift = line.rstrip().split(",")
            shifts.append((float(shift[0]), float(shift[1])))
    data.close()
    return shifts


def load_max_hours(filename):
    """
    Load data about maximum hours that can be scheduled for each day.
    :param filename: name of the file storing the data
    :return: a dictionary that maps a day of a week to the max hours
    """
    max_hours = {}
    data = open("data/" + filename)
    for line in data:
        if not line.startswith("#"):
            max_hours[line.split(":")[0]] = line.split(":")[1]
    data.close()
    return max_hours


def get_length(shift):
    """
    Return the duration of a shift.
    :param shift: a shift represented by a tuple in the form: (start_time, end_time)
    :return: duration of the shift
    """
    return shift[1] - shift[0]


def is_on(shift, time):
    """
    Check if the given shift is on at the given time.
    :param shift: a tuple representing a shift
    :param time: a number representing time
    :return: 1 if the shift is on at the given time, 0 otherwise
    """
    if shift[0] <= time < shift[1]:
        return 1
    else:
        return 0


def separate_hour(hour):
    """
    Separate opening hour of a station represented by a tuple to a list of times with intervals of 15 mins
    :param hour: opening hour to be separated
    :return: a list of times separated by 15 mins during the opening hour
    """
    result = []
    time = hour[0]
    end = hour[1]
    while time < end:
        result.append(time)
        time += 0.25
    return result


def float_to_time(num):
    """
    Convert a float representation of time to standard representation of time.
    :param num: float representation of time
    :return: standard representation of time in string
    """
    hour = str(int(num))
    minute = str(int((num - int(num)) * 60))
    if minute == "0":
        return hour + ":00"
    return hour + ":" + minute


def between_one_and_four(station, day, name, model_name, constrs):
    """
    Add constraints to a model restricting that at any time, separated by 15 mins,
    every station has at least 1 and at most 4 workers during opening hours
    :param station: station of interest
    :param day: day of a week of interest
    :param name: name of the station in lowercase
    :param model_name: name of the model to add constraints on
    """
    for hour in station.get_hours(day):
        for time in separate_hour(hour):
            is_on_list = list(map(lambda x: is_on(x, time), station.get_shifts()))
            constr_builder = ""
            for i in range(len(is_on_list)):
                if is_on_list[i] == 1:
                    constr_builder += name + "_x" + str(i) + " + "
            constrs.add("%s.addConstr(%s >= 1)" % (model_name, constr_builder[:-3]))
            constrs.add("%s.addConstr(%s <= 4)" % (model_name, constr_builder[:-3]))


def opening_closing_prep(station, day, name, model_name, constrs):
    """
    Set constraints that every station need exactly one student 15 mins before
    opening and 30 mins after closing. Quench is specially that it need one
    student 1 hour before opening.
    :param station: station of interest
    :param day: day of a week of interest
    :param name: name of the station in lowercase
    :param model_name: name of the model to add constraints on
    :return:
    """
    for hour in station.get_hours(day):
        start = hour[0]
        end = hour[1]
        shifts = station.get_shifts()
        for i in range(len(shifts)):
            if name == "quench":
                if shifts[i][0] == start - 1:
                    constrs.add("%s.addConstr(%s_x%d == 1)" % (model_name, name ,i))
            elif name == "market":
                if shifts[i][0] == 10.5:
                    constrs.add("%s.addConstr(%s_x%d == 1)" % (model_name, name, i))
            else:
                if shifts[i][0] == start - 0.25:
                    constrs.add("%s.addConstr(%s_x%d == 1)" % (model_name, name, i))
            if shifts[i][1] == end + 0.5:
                constrs.add("%s.addConstr(%s_x%d == 1)" % (model_name, name, i))


def total_hours(stations, max_hours):
    """
    Add constraint on maximum hours can be scheduled on that day.
    :param stations: a distionary of all station, mapping name of a station
                     to the object storing data about that station
    :param max_hours: a dictionary mapping day of a week to maximum
                      hours can be scheduled on that day
    :return: string representation of the sum of all length of shifts
    """
    constr_builder = ""
    j = 0
    for name in stations:
        shifts = stations[name].get_shifts()
        for i in range(len(shifts)):
            duration = get_length(shifts[i])
            constr_builder += name + "_x" + str(i) + " * " + str(duration) + " + "
    return constr_builder[:-3]


def build_obj_func(station, name, day, customer_count, prob):
    """
    Build up a string representation of the final objective function.
    :param station: station that we are considering
    :param name: name of the station in lowercase
    :param day: day of a week
    :param customer_count: number of customers coming in at any given time
    :param prob: probability of a customer going to any station an any time
    :return: string representation of the final objective function
    """
    hours = station.get_hours(day)
    obj_func = ""
    for hour in hours:
        times = separate_hour(hour)
        for time in times:
            cur_customer_num = customer_count[time]
            cur_prob = prob[time]
            is_on_list = list(map(lambda x: is_on(x, time), station.get_shifts()))
            working_shift_builder = ""
            for i in range(len(is_on_list)):
                if is_on_list[i] == 1:
                    working_shift_builder += name + "_x" + str(i) + " + "
            speed = station.get_speed().replace("x", "(" + str(working_shift_builder[:-3]) + ")")
            # number of customers is average number of that day
            obj_func_builder = str(math.ceil(cur_customer_num * cur_prob)) + " * " + "(" + speed + ")" + " + "
            obj_func += obj_func_builder
    return obj_func


# -------- Main Optimization Functions --------- #
def optimizer(stations, day, model_name, max_hours, all_shifts):
    """
    Add all variables and relative constraints to a given model
    :param stations: a map from name to station object for all stations
    :param day: day of a week of interest
    :param model_name: name of the model in string
    :param max_hours: a map from day to the max hours can be scheduled on that day
    :param all_shifts: map from index to a shift, initially empty to be filled
    """
    # load in customer count for a day
    customer_count = load_customer_count(day + "_count.txt")

    # load in probability for each station on a day
    select_prob = {}
    seared_prob = {}
    market_prob = {}
    plate_prob = {}
    noodle_prob = {}
    quench_prob = {}
    load_prob([select_prob, seared_prob, market_prob, plate_prob, noodle_prob, quench_prob],
              day + "_prob.csv")
    prob_dict = {"select": select_prob,
                 "seared": seared_prob,
                 "market": market_prob,
                 "plate": plate_prob,
                 "noodle": noodle_prob,
                 "quench": quench_prob}

    # Add variables for all shifts on all stations
    counter = 0
    for name in stations:
        for i in range(stations[name].get_shift_num()):
            all_shifts[counter] = ((name + "_x%d" % i), stations[name].get_shifts()[i])
            counter += 1
            exec ("%s_x%d = %s.addVar(vtype=GRB.INTEGER)" % (name, i, model_name))

    # Add non-negativity constraints on all stations
    # Add constraints such that at any time, separated by 15 minutes,
    # every station has between 1 and 4 workers during opening hours
    # Add constrains such that every station has worker to do opening
    # preparation and closing cleaning
    # Build up string representing the objective function
    constrs = Set([])
    obj_func = ""
    for name in stations:
        for i in range(stations[name].get_shift_num()):
            eval("%s.addConstr(%s_x%d >= 0)" % (model_name, name, i))
        if stations[name].get_hours(day) is not None:
            between_one_and_four(stations[name], day, name, model_name, constrs)
            opening_closing_prep(stations[name], day, name, model_name, constrs)
            obj_func += build_obj_func(stations[name], name, day, customer_count, prob_dict[name])

    # Market opens later on Saturday and Sunday
    if day == "Saturday" or day == "Sunday":
        constrs.add("%s.addConstr(market_x0 == 0)" % model_name)
    for constr in constrs:
        eval(constr)

    # Add constraint on total hours on a day
    eval("%s.addConstr(%s <= %s)" % (model_name, total_hours(stations, max_hours), max_hours[day]))

    # Add objective function
    eval("%s.setObjective(%s, GRB.MINIMIZE)" % (model_name, obj_func[:-3]))


def interpret_result(model, day, all_shifts):
    """
    Interpret the result and write the result to text file.
    :param model: model with variables and constraints added
    :param day: day of a week of interest
    :param all_shifts: map from index to a shift
    """
    model.optimize()
    output = open("data/" + day + "_output.txt", "w")
    hours_sum = 0
    for var in model.getVars():
        index = int(var.varName[1:])
        value = int(var.x)
        cur_shift = all_shifts[index]
        name = cur_shift[0].split("_")[0]
        time = float_to_time(cur_shift[1][0]) + " - " + float_to_time(cur_shift[1][1]) + "\n"
        hours_sum += get_length(cur_shift[1]) * value
        for i in range(value):
            output.write(name.capitalize() + ": " + time)
    output.write("Total Hours: " + str(hours_sum))


# ---------- Setting up Data ---------- #
# Setting up data for all stations
plate = Station("Plate", load_hours("plate_hours.txt"),
                load_shifts("plate_shifts.txt"), "-0.2 * x + 1")
# -0.2 * x + 1.2
quench = Station("Quench", load_hours("quench_hours.txt"),
                load_shifts("quench_shifts.txt"), "-0.2 * x + 1.6")
noodle = Station("Noodle", load_hours("noodle_hours.txt"),
                load_shifts("noodle_shifts.txt"), "-0.2 * x + 1.2")
select = Station("Select", load_hours("select_hours.txt"),
                load_shifts("select_shifts.txt"), "-0.2 * x + 1")
market = Station("Market", load_hours("market_hours.txt"),
                load_shifts("market_shifts.txt"), "-0.2 * x + 1.1")
# -0.1 * x + 1.5
seared = Station("Seared", load_hours("seared_hours.txt"),
                load_shifts("seared_shifts.txt"), "-0.2 * x + 1")
# -0.2 * x + 1.6

# A map from name of the station in string to the station's data in an object
all_stations = {"plate": plate, "quench": quench,
                "noodle": noodle, "select": select,
                "market": market, "seared": seared}

# ---------- Start Optimization ---------- #
# Optimize for Monday
m_model = Model()
all_shifts = {}
optimizer(all_stations, "Monday", "m_model", load_max_hours("max_hours.txt"), all_shifts)
interpret_result(m_model, "Monday", all_shifts)

# Optimize for Tuesday
t_model = Model()
optimizer(all_stations, "Tuesday", "t_model", load_max_hours("max_hours.txt"), all_shifts)
interpret_result(t_model, "Tuesday", all_shifts)

# Optimize for Wednesday
w_model = Model()
optimizer(all_stations, "Wednesday", "w_model", load_max_hours("max_hours.txt"), all_shifts)
interpret_result(w_model, "Wednesday", all_shifts)

# Optimize for Thursday
th_model = Model()
optimizer(all_stations, "Thursday", "th_model", load_max_hours("max_hours.txt"), all_shifts)
interpret_result(th_model, "Thursday", all_shifts)

# Optimize for Friday
f_model = Model()
optimizer(all_stations, "Friday", "f_model", load_max_hours("max_hours.txt"), all_shifts)
interpret_result(f_model, "Friday", all_shifts)

# Optimize for Saturday
sa_model = Model()
optimizer(all_stations, "Saturday", "sa_model", load_max_hours("max_hours.txt"), all_shifts)
interpret_result(sa_model, "Saturday", all_shifts)

# Optimize for Friday
su_model = Model()
optimizer(all_stations, "Sunday", "su_model", load_max_hours("max_hours.txt"), all_shifts)
interpret_result(su_model, "Sunday", all_shifts)