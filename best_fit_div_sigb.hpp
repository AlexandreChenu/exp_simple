#ifndef BEST_FIT_DIV_SB
#define BEST_FIT_DIV_SB

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <sferes/stat/stat.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/stat/best_fit.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "fit_behav.hpp"

namespace sferes {
  namespace stat {
    SFERES_STAT(BestFitDivSb, Stat) {
    public:
      template<typename E>
      void refresh(const E& ea) {
        assert(!ea.pop().empty());
        _best = *std::max_element(ea.pop().begin(), ea.pop().end(), fit::compare_max());


        this->_create_log_file(ea, "bestfit.dat");
        if (ea.dump_enabled())
          (*this->_log_file) << ea.gen() << " " << ea.nb_evals() << " " << _best->fit().value() << std::endl;

        //change it to depend from params 
        if (_cnt%Params::pop::dump_period == 0){ //for each dump period

          Eigen::MatrixXd zones_cnt = Eigen::MatrixXd::Zero(101,101);
          Eigen::Vector3d target;

          target = {-0.211234, 0.59688,0.0};

          std::cout << "pop size: " << ea.pop().size() << std::endl;

          for (int i = 0; i < ea.pop().size(); ++i){
                zones_cnt += run_simu(*ea.pop()[i], target);
                }

          int sum_zones = 0;

          for (float i = 0; i < 101; i+=1){
            for (float j = 0; j < 101; j+=1){
              if (zones_cnt(i,j) != 0)
                sum_zones += 1;
              }}

          double novelty_score = sum_zones;
          double novelty_score_n = novelty_score /(100*100);

          std::cout << "novelty score is: " << novelty_score << std::endl;

          std::cout << "normalized novelty score is: " << novelty_score_n <<  std::endl;

          _nov_scores.push_back(novelty_score);


          std::cout << "dump period" << std::endl;}

        _cnt += 1;

        if (_cnt == Params::pop::nb_gen){

          std::cout << "Saving novelty scores" << std::endl;

          std::string filename_out = ea.res_dir() + "novelty_gte.txt"; //file containing samples
          //std::string filename_out = "/git/sferes2/results_sigb_nov/novelty_gte.txt";
          std::ofstream out_file; 
          out_file.open(filename_out);

          if (!out_file) { //quick check to see if the file is open
            std::cout << "Unable to open file " << filename_out;
            exit(1);}   // call system to stop

          for (int i = 0; i < _nov_scores.size(); i++){
            out_file << _nov_scores[i] << std::endl;
          }
          out_file.close();
        }
      }

      void show(std::ostream& os, size_t k) {
        _best->develop();
        _best->show(os);
        _best->fit().set_mode(fit::mode::view);
        _best->fit().eval(*_best);
      }
      const boost::shared_ptr<Phen> best() const {
        return _best;
      }
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_NVP(_best);
      }

          template <typename T>
    Eigen::MatrixXd run_simu(T & model, Eigen::Vector3d target) { 

        //std::cout << "start initialization" << std::endl;

        Eigen::MatrixXd work_zones_cnt = Eigen::MatrixXd::Zero(101,101);

        //init variables
        double _vmax = 1;
        double _delta_t = 0.1;
        double _t_max = 10; //TMax guidé poto
        Eigen::Vector3d robot_angles;

        Eigen::Vector3d prev_pos; //compute previous position
        Eigen::Vector3d pos_init;

        robot_angles = {0,M_PI,M_PI}; //init everytime at the same place

        double radius;
        double theta;

        model.develop();

        double dist = 0;

        //get gripper's position
        prev_pos = forward_model(robot_angles);
        pos_init = forward_model(robot_angles);

        std::vector<float> inputs(5);

        for (int t=0; t< _t_max/_delta_t; ++t){
              
              inputs[0] = target[0] - prev_pos[0]; //get side distance to target
              inputs[1] = target[1] - prev_pos[1]; //get front distance to target
              inputs[2] = robot_angles[0];
              inputs[3] = robot_angles[1];
              inputs[4] = robot_angles[2];


              for (int j = 0; j < model.gen().get_depth() + 1; ++j) //In case of FFNN
                model.nn().step(inputs);
              
              Eigen::Vector3d output;
              for (int indx = 0; indx < 3; ++indx){
                output[indx] = 2*(model.nn().get_outf(indx) - 0.5)*_vmax; //Remap to a speed between -v_max and v_max (speed is saturated)
                robot_angles[indx] += output[indx]*_delta_t; //Compute new angles
              }

              //Eigen::Vector3d new_pos;
              prev_pos = forward_model(robot_angles); //remplacer pour ne pas l'appeler deux fois


              int x_int = prev_pos[0]*100;
              int y_int = prev_pos[1]*100;

              int indx_X =0;
              int indx_Y =0;
              
              if (x_int %2 !=0)
                  indx_X = (x_int + 100)/2;
              
              else 
                  indx_X = (x_int + 101)/2 ;
              
              if (y_int %2 !=0)
                  indx_Y = (y_int + 100)/2;
              
              else 
                  indx_Y = (y_int + 101)/2;
          
              work_zones_cnt(indx_X,indx_Y) ++;

            }

        Eigen::Vector3d final_pos; 
        final_pos = forward_model(robot_angles);

        double out_fit;

        if (sqrt(square(target.array() - final_pos.array()).sum()) < 0.02){
          std::cout << "task successful" << std::endl;
          return work_zones_cnt;} // -> 1

        else {
          return Eigen::MatrixXd::Zero(101,101);} // -> 0
    }


    Eigen::Vector3d forward_model(Eigen::VectorXd a){
    
    Eigen::VectorXd _l_arm=Eigen::VectorXd::Ones(a.size()+1);
    _l_arm(0)=0;
    _l_arm = _l_arm/_l_arm.sum();

    Eigen::Matrix4d mat=Eigen::Matrix4d::Identity(4,4);

    for(size_t i=0;i<a.size();i++){

      Eigen::Matrix4d submat;
      submat<<cos(a(i)), -sin(a(i)), 0, _l_arm(i), sin(a(i)), cos(a(i)), 0 , 0, 0, 0, 1, 0, 0, 0, 0, 1;
      mat=mat*submat;
    }
    
    Eigen::Matrix4d submat;
    submat<<1, 0, 0, _l_arm(a.size()), 0, 1, 0 , 0, 0, 0, 1, 0, 0, 0, 0, 1;
    mat=mat*submat;
    Eigen::VectorXd v=mat*Eigen::Vector4d(0,0,0,1);

    return v.head(3);

  }

    protected:
      int _cnt = 0; //not sure if it is useful
      boost::shared_ptr<Phen> _best;
      int _nbest = 3;
      std::vector<double> _nov_scores;
    };
  }
}
#endif
