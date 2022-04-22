#include <iostream>
#include <numbers>
#include <cmath>
#include <fstream>

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"

//#include <opencv2/opencv.hpp>

typedef ceres::Grid2D<double,1,true,true> ImageGrid;
typedef ceres::BiCubicInterpolator<ImageGrid> Interpolator;

class ImageDiffFunctor
{
    typedef ceres::AutoDiffCostFunction<ImageDiffFunctor, 1, 2> ImageDiffFunction;
    typedef ceres::NumericDiffCostFunction<ImageDiffFunctor, ceres::NumericDiffMethodType::CENTRAL, 1, 2> ImageDiffFunctionNumeric;

public:
    ImageDiffFunctor(
            int width,
            int height,
            int xPos,
            int yPos,
            double* img0,
            Interpolator& img1Interpolator
    ) :
            m_width(width),
            m_height(height),
            m_xPos(xPos),
            m_yPos(yPos),
            m_img0(img0),
            m_img1Interpolator(img1Interpolator)
    {
    }

    static ceres::CostFunction* create(
            int width,
            int height,
            int xPos,
            int yPos,
            double* img0,
            Interpolator& img1Interpolator
    )
    {
        auto functor = new ImageDiffFunctor(width, height, xPos, yPos, img0, img1Interpolator);
        // Auto Diff
        return new ImageDiffFunction(functor);
        // Numeric Diff
//        return new ImageDiffFunctionNumeric(functor, ceres::TAKE_OWNERSHIP);
    }

    template <class T> inline
    bool operator()(const T* const t, T* residuals) const
    {
        bool xValid = T(0) <= T(m_xPos) + t[0] && T(m_xPos) + t[0] < T(m_width);
        bool yValid = T(0) <= T(m_yPos) + t[1] && T(m_yPos) + t[1] < T(m_height);

        T pe = T(0);
        if( xValid && yValid )
        {
            T i1;
            m_img1Interpolator.Evaluate(T(m_yPos)+t[1], T(m_xPos)+t[0], &i1);

            double i0 = *(m_img0 + m_yPos * m_width + m_xPos);
            pe = T(i0) - i1;
        }

        residuals[0] = pe;
        return true;
    }

private:
    const int m_width;
    const int m_height;
    const int m_xPos;
    const int m_yPos;
    double* m_img0;

    Interpolator& m_img1Interpolator;
};

void create_czp_image(int w, int h, double cx, double cy, double* pBuf)
{
    const double A = 125.5;
    const double B = 128;
    const double theta = std::numbers::pi / 2.0;

    for( int y = 0 ; y < h ; ++y)
    {
        for( int x = 0 ; x < w ; ++x)
        {
            double d = std::pow(x-cx,2.0) / cx + std::pow(y-cy,2.0) / cy;
            *(pBuf + y * w + x) = A * std::sin(std::numbers::pi / 2.0 * d + theta) + B;
        }
    }
}

//void showImage(int w, int h, double* pBuf, std::string title="")
//{
//    cv::Mat cv_mat(h, w,CV_64F, pBuf);
//
//    cv::Mat converted;
//    cv_mat.convertTo(converted, CV_8U);
//
//    cv::imshow(title, converted);
//    cv::waitKey(1);
//}

void optimize(int w, int h, double* pImg0, double* pImg1, double tx, double ty)
{
    ceres::Problem problem;

    ImageGrid grid(pImg1, 0, h, 0, w);
    Interpolator img1Interp(grid);

    std::vector<double> translate{tx, ty};

    std::cout << std::endl << "AddResidualBlock ..." << std::endl;
    for( int y = 0 ; y < h ; ++y)
    {
        for( int x = 0 ; x < w ; ++x)
        {
            auto cost = ImageDiffFunctor::create( w, h, x, y, pImg0, img1Interp);
            problem.AddResidualBlock(cost, nullptr, translate.data());
        }
    }

    //
    // optimize
    //
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
//    options.check_gradients = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;
    std::cout << summary.FullReport() << std::endl;
    std::cout << "solution : x = " << translate[0] << ", y = " << translate[1] << std::endl;

    //
    // evaluate
    //
    double cost;
    std::vector<double> residuals, gradient;
    ceres::CRSMatrix jacobian;
    auto opt = ceres::Problem::EvaluateOptions();
    problem.Evaluate(opt, &cost, &residuals, &gradient, &jacobian);

    std::cout << "size of residuals = " << residuals.size() << std::endl;
    std::cout << "size of gradient = " << gradient.size() << std::endl;
    {
        std::ofstream ofs("residual.txt");
        for (auto i=0; i < residuals.size() ; ++i) {
            ofs << residuals[i] << std::endl ;
        }
    }
    {
        std::ofstream ofs("gradient.txt");
        for (auto i=0; i < gradient.size() ; ++i) {
            ofs << gradient[i] << std::endl ;
        }
    }

}

int main() {
    // image size
    const int width = 320;
    const int height = 240;
    // Img1 offset
    const double offsetX = 5;
    const double offsetY = 0;
    // position of residual block
    const int xPos = int(round(width-1.0)/2.0) + 100;
    const int yPos = int(round(height-1.0)/2.0) + 100;
    // initial parameter(translation)
    const double tx = 5;
    const double ty = 0;

    //
    // setup images
    //
    auto pImg0 = new double[width * height];
    auto pImg1 = new double[width * height];

    double cx = (width-1.0) / 2.0;
    double cy = (height-1.0) / 2.0;
    create_czp_image(width, height, cx, cy,pImg0);
    create_czp_image(width, height, cx+offsetX, cy+offsetY,pImg1);

//    showImage(width, height, pImg0, "Img0");
//    showImage(width, height, pImg1, "Img1");

    //
    // GradientCheck
    //
    ImageGrid grid(pImg1, 0, height, 0, width);
    Interpolator img1Interp(grid);

    auto cost = ImageDiffFunctor::create(
            width, height,
            xPos, yPos,
            pImg0, img1Interp);

    std::vector<const ceres::Manifold*> manifolds;
    manifolds.push_back(nullptr);

    ceres::NumericDiffOptions options;
    ceres::GradientChecker checker(cost, &manifolds, options);

    std::vector<double> translate{tx, ty};
    std::vector<double*> parameter_blocks;
    parameter_blocks.push_back(translate.data());

    ceres::GradientChecker::ProbeResults results;
    if (!checker.Probe(parameter_blocks.data(), 1e-8, &results)) {
        std::cout << "An error has occurred:\n" << results.error_log;
    }

    std::cout << std::endl << "GradientChecker ProbeResults" << std::endl;
    std::cout << "return_value : " <<  results.return_value << std::endl;
    std::cout << "maximum_relative_error : " <<  results.maximum_relative_error << std::endl;
    std::cout << "residuals : " <<  results.residuals(0) << std::endl;
    std::cout << "jacobians : " <<  results.jacobians[0](0) << ", " <<  results.jacobians[0](1) << std::endl;
    std::cout << "local_jacobians : " <<  results.local_jacobians[0](0) << ", " <<  results.local_jacobians[0](1) << std::endl;
    std::cout << "numeric_jacobians : " <<  results.numeric_jacobians[0](0) << ", " <<  results.numeric_jacobians[0](1) << std::endl;
    std::cout << "local_numeric_jacobians : " <<  results.local_numeric_jacobians[0](0) << ", " <<  results.local_numeric_jacobians[0](1) << std::endl;

//    cv::waitKey(0);

    //
    // optimize
    //
    optimize(width, height, pImg0, pImg1, tx, ty);

    delete [] pImg0;
    delete [] pImg1;

    return 0;
}
